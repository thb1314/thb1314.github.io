---
comments: true
---

# 分布式训练场景下ModelEMA的优化

> 本文写于2024年9月9号晚22点

## 一、前言

有一天白天喝茶饮料喝多了，怎么也睡不着。于是尝试想一想ModelEMA的分布式优化版本，由于不满足于这种系统实现上的优化，手推公式一顿近似化简想把ModelEMA的行为放到优化器中，结果第二天一早实现后Loss NaN。

就这样拖了一周，再到后面重新思考ModelEMA的分布式实现，刚好看到torch官方zero2的源码，所以将其参数平均分配到各个rank的算法移植过来，再加上将不连续内存合并的优化，效果真的很惊艳。

最开始的想法启发于zero2，即针对EMA运算，每个计算卡分别存储和计算模型整体参数的一小部分，在模型评估阶段再对所有参数进行all gather操作。

以8卡分布式数据并行为例，记原始yolov5 EMA更新操作单个training step 55ms，开启分布式EMA后达到7ms（接近55/8），开启内存合并后降为0.5ms，速度提升了100多倍！

由于小模型参数量不大，所以没有计算节省的参数量。

> 本文相关代码开源在
> <https://github.com/thb1314/distributed_modelema>

## 二、实现原理

ModelEMA可以有效地缓解过拟合问题并提升泛化性，在自监督学习任务，比如MoCO、DINO、BYOL等，ModelEMA在该类训练任务中起重要作用。

ModelEMA的更新过程如下

$$
\zeta_{t+1}=\rho \zeta_t+(1-\rho) \theta_t
$$

其中 $\zeta$ 表示EMA模型参数（初始化为模型参数）， $\theta_t$ 表示optimizer更新后的模型参数。

启发于ZERO2，针对上述运算，假设总共的计算单元有world_size个，我们可以将原版模型的参数分为world_size组，每组参数分别在各自的计算单元中执行EMA操作，最后在**有需要的时候**再将参数all gather到所有机器中。

那么什么时候是“有需要的时候”呢？即“需要采用EMA后的模型做评估的时候”

## 三、实现细节

若要实现分布式版本EMA，参数分配算法和参数同步直观重要。本文提出的实现，首先将state_dict中的参数转换为parameter中的参数，接着采用参数分配算法将parameter划分到各个rank中，然后在有必要的时候执行EMA同步操作，同时可以有选择性地采用Tensor合并算法对EMA过程进行优化。

### 3.1 参数接口转换

由于state_dict中参数都是detach后的，如下代码片段实现将`model.state_dict()`中的参数转换为（来不及解释了，看如下源码吧）




```python
# 收集原模型 state_dict
self._ori_state_dict:Dict[str, nn.Parameter] = de_parallel(model).state_dict()
# replace to original parameter
# 原模型 state_dict 与 pamameter和buffer中的参数data_ptr相同
ori_param_dict = {param.data_ptr():param for param in de_parallel(model).parameters()}
ori_param_dict.update({buffer.data_ptr():buffer for buffer in de_parallel(model).buffers()})

# 统计不需要ema的参数
self._no_need_ema_dict = dict()
for name, param in self._ori_state_dict.items():
    if param.data_ptr() in ori_param_dict and param.dtype.is_floating_point:
        self._ori_state_dict[name] = ori_param_dict[param.data_ptr()]
        else:
            self._no_need_ema_dict[name] = param
            for rm_name in self._no_need_ema_dict:
                self._ori_state_dict.pop(rm_name)
```



### 3.2 参数分配算法

参考ZeroRedundancyOptimizer的实现，partition_parameters 方法会将参数进行分区，根据参数大小（而不是使用顺序）以排序贪婪（sorted-greedy）算法来对优化器状态进行分片，在每个rank中打包一些参数，这样每个参数都属于一个rank，不在ranks之间划分。分区是任意的，可能与参数注册或使用顺序不匹配。这是为了确保每个rank具有几乎相同大小的显存。




```python
def partition_parameters(self) -> List[Dict[str, nn.Parameter]]:
    r"""
        Partitions parameters across distributed data parallel ranks.

        Returns:
            a list of ``param_groups`` (which is a list of dict) where each
            element of the list contains the param_groups for a rank. Element 0
            corresponds to rank 0, etc. We need all the ranks for the broadcast
            inside ``get_model_state_dict()``.
        """
    if len(self._partition_parameters_cache) == 0:
        self._partition_parameters_cache = [dict() for _ in range(self.world_size)]
        # 生成一个数组，用来记录每个rank的大小，一共有world size个rank
        sizes = [0] * self.world_size

        # 遍历参数组
        param_lists: List[List[Tuple[str, nn.Parameter]]] = [list() for _ in range(self.world_size)]
            for name, param in self._ori_state_dict.items():
                # add this param to rank with smallest size
                # 找到最小的那个rank
                rank = sizes.index(min(sizes))
                # 把参数放到最小rank之中
                param_lists[rank].append((name, param))
                # 增加rank的大小
                sizes[rank] += param.numel()

                # 遍历list
                for rank, param_tuple_list in enumerate(param_lists):
                    for name, param in param_tuple_list:
                        self._partition_parameters_cache[rank][name] = param

                        return self._partition_parameters_cache
```




这里就分区好了，最终返回一个param_groups 的列表（这是一个dict列表），列表的每个元素都包含一个rank的param_groups，比如元素0对应于rank 0，每个rank的group的参数有差不多大小。

### 3.3 同步EMA参数

需要注意的是`get_model_state_dict`需要每个rank都得执行，通过判断参数是在当前rank下还是其他rank下来获取源头的rank地址，之后执行`dist.broadcast`来广播tensor到其他rank。




```python
    def get_model_state_dict(self, strict=True):
        ema_state_dict = OrderedDict()
        ori_state_dict = OrderedDict()
        handles = []

        for key in self._ori_state_dict:
            if key in self._no_need_ema_dict:
                if not strict:
                    continue
                # adopt its original reference
                ema_state_dict[key] = self._no_need_ema_dict[key]
                ori_state_dict[key] = self._no_need_ema_dict[key]
            elif key in self._ori_state_dict:
                # send parameters
                if key in self._cur_rank_param:
                    param_value = self._cur_rank_param[key]
                    ema_state_dict[key] = param_value
                    ori_state_dict[key] = self._ori_cur_rank_param[key].detach().clone()
                    if self.world_size > 1:
                        handles.append(dist.broadcast(tensor=param_value.data, src=self.rank, group=self.group, async_op=True))
                elif key in self._other_rank_param:
                    param_value = self._other_rank_param[key]
                    src_rank = self._other_param2rank[param_value]
                    ori_state_dict[key] = param_value.detach().clone()
                    param_value = param_value.detach().clone()
                    ema_state_dict[key] = param_value
                    if self.world_size > 1:
                        handles.append(dist.broadcast(tensor=param_value.data, src=src_rank, group=self.group, async_op=True))
                else:
                    raise RuntimeError(f"{key} not in parameter list")
            else:
                raise RuntimeError(f"{key} not in parameter list")

        _ = list(map(lambda x: x.wait(), handles))
        return ema_state_dict, ori_state_dict
```




这里需要注意的是，broadcast操作是异步的。

### 3.4 Tensor合并

如果设置了`parameters_as_bucket_view`，则调用建立若干buffer。同样设备上同样rank的张量合并一个buffer，这里需要注意的是个别处理的字节对齐问题，本文实现的是8字节对齐版本。




```python
        if parameters_as_bucket_view and self._ori_cur_rank_param:
            device = next(iter(self._ori_cur_rank_param.values())).device
            dtype = next(iter(self._ori_cur_rank_param.values())).dtype
            buffer_size = 0
            # 8 bytes aligned
            grid_size = 8 // item_size_dict[dtype]

            # 统计参数排序信息
            for key, param in self._ori_cur_rank_param.items():
                offset_start = buffer_size
                buffer_size += (param.numel() + grid_size - 1) // grid_size * grid_size
                self._bucket_data_info_dict[key] = {
                    "offset_start": offset_start,
                    "offset_end": buffer_size,
                    "real_size": param.numel()
                }
               # 初始化 bucket 参数大小
            bucket = nn.Parameter(torch.empty(buffer_size, dtype=dtype, device=device), requires_grad=False)
            self._ori_cur_rank_bucket = bucket

            # 根据偏移 copy 原始数据
            for key, param in self._ori_cur_rank_param.items():
                data_info_dict = self._bucket_data_info_dict[key]
                offset = data_info_dict['offset_start']
                offset_next = offset + data_info_dict['real_size']
                bucket[offset:offset_next].copy_(param.data.flatten(), non_blocking=False)
                param.data = bucket[offset:offset_next].view_as(param.data)

        self._cur_rank_param:Dict[str, nn.Parameter] = dict()
        self._cur_rank_bucket:Optional[nn.Parameter] = None
        if self._ori_cur_rank_bucket is not None:
            self._cur_rank_bucket = self._ori_cur_rank_bucket.detach().clone()

        # 如果设置了bucket则将param.data指向buffer中的区域，从而param的更新会自动更新到buffer
        for name, param in self._ori_cur_rank_param.items():
            param = param.detach().clone()
            self._cur_rank_param[name] = param
            param.requires_grad_(False)
            if self._cur_rank_bucket is not None:
                data_info_dict = self._bucket_data_info_dict[name]
                offset = data_info_dict["offset_start"]
                offset_next = offset + data_info_dict["real_size"]
                param.data = self._cur_rank_bucket[offset:offset_next].view_as(param.data)
```




开启Tensor合并后，大大提高的程序的并行化程度，从而获得极致的加速效果。

## 总结

ModelEMA的分布式实现不算很难，举一反一，很朴素的想法。

尽管思路简单，但是效果真的很惊艳。

## 参考文献与链接

* [PyTorch 分布式之 ZeroRedundancyOptimizer](https://www.cnblogs.com/rossiXYZ/p/15782054.html)

