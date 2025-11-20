---
comments: true
---

# Numpy和Eigen—Python与C++中的矩阵运算库的联系

> 本文写于2024年9月17日晚十点

## 一、需求背景

在一个纯C/C++的项目中的项目中，算法开发人员在CPU上习惯采用python中的numpy库编写了相关优化算法，但工程要求需要将python实现的算法以C/C++语言重新表达以方便在各个硬件平台上的编译移植。

由此，引出了该项技术：Python中的矩阵运算库numpy与C++中的矩阵元素库-Eigen之间的转换。

## 二、相关库介绍

NumPy（Numerical Python的简称）是一个开源的Python科学计算库，用于处理大型多维数组和矩阵，提供了大量的数学函数来操作这些数组。以下是NumPy的一些主要特点：  

1. **强大的数组对象**：NumPy提供了一个强大的n维数组对象（即`ndarray`），它是一个用于存储同类型数据的多维容器。这种数组比Python原生的列表更加高效和方便。  
2. **广播功能**：NumPy的广播功能允许不同形状的数组在运算时进行自动扩展，使得代码更加简洁。  
3. **数学函数库**：NumPy提供了大量的数学函数，可以用于执行各种数学运算，如线性代数、傅里叶变换、概率分布等。  
4. **高效的内存使用**：NumPy的数组在内存中是连续存储的，这有助于提高数据处理的效率。  
5. **成熟的生态系统**：NumPy是Python科学计算生态系统的基础，许多其他科学计算库（如Pandas、SciPy、Matplotlib等）都依赖于NumPy。  

NumPy广泛应用于数据分析、机器学习、图像处理、物理学模拟等领域，是Python科学计算不可或缺的工具之一。

---

Eigen是一个高级的C++库，用于线性代数、矩阵和向量运算，数值解算以及相关的算法。它是由Benjamin Schindler、Gaël Guennebaud和其他贡献者共同开发的，旨在提供一个易于使用、高效且灵活的API来处理数值计算问题。以下是Eigen的一些主要特点：  

1. **矩阵和向量的表示**：Eigen提供了丰富的数据结构来表示矩阵和向量，支持动态和固定大小的类型，以及特殊类型的矩阵，如对角矩阵、三对角矩阵等。
2. **直观的接口**：Eigen的接口设计得很直观，使得编写矩阵运算的代码就像在纸上写数学公式一样简单。
3. **高效的性能**：Eigen经过高度优化，能够充分利用现代CPU的指令集，如SSE和AVX，以提供快速的矩阵运算。
4. **完全的模板化**：Eigen是一个完全模板化的库，这意味着它可以与任何数值类型一起使用，包括标准的内置类型、用户定义的类型以及第三方库中的类型。
5. **支持复杂的数学运算**：Eigen不仅支持基本的线性代数运算，还支持更复杂的数学运算，如特征值分解、奇异值分解、矩阵求逆、最小二乘求解等。
6. **易于集成**：Eigen可以很容易地集成到现有的C++项目中，只需要包含相应的头文件即可。
7. **跨平台**：Eigen是跨平台的，可以在多种操作系统上编译和使用，包括Linux、Windows和macOS。

Eigen广泛应用于科学研究、工程计算、机器学习、计算机视觉等领域，是C++开发者进行数值计算的一个强大工具。由于其高效的性能和易用的接口，Eigen在需要高性能数值计算的项目中非常受欢迎。

> 对上述两者的相互转换要求开发者熟悉矩阵运算的表达和化简，以及对于numpy和Eigen，python、C++等都比较熟悉。

## 三、相关转换

### 3.1 矩阵数据读取内存的转换

Eigen默认是列优先（column-major）的存储顺序。这意味着在内存中，矩阵的列是连续存储的。这与MATLAB和Fortran中的存储顺序相同，但与C/C++默认的行优先（row-major）存储顺序不同。

在Eigen中，如果你想要使用行优先存储顺序，可以采用如下代码定义行优先的类：


```cpp
#include <Eigen/Dense>

namespace Eigen {
    using MatrixXiRowMajor = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
};
```

NumPy默认使用行优先（row-major）的存储顺序，保存npy文件时默认也是行优先存储，这意味着在内存中，矩阵的行是连续存储的，这与C语言中的数组存储顺序是一致的。

如果想实现numpy中的npy文件以正常的Eigen形式表达，方式之一是采用列优先方式读取，然后转置。

在列优先的设置下，先将行和列的设置互换，


```cpp
Eigen::MatrixXfRowMajor load_conf_npy(const std::string& npy_path) {
    cnpy::NpyArray conf_1_npy = cnpy::npy_load(npy_path);
    int col = conf_1_npy.shape.back();
    int row = 1;
    for(int i = 0; i < (int)conf_1_npy.shape.size() - 1; ++i) {
        row *= conf_1_npy.shape[i];
    }
    return Eigen::Map<Eigen::MatrixXf, Eigen::ColMajor>(conf_1_npy.data<float>(), col, row).transpose();
}
```




举个栗子：

原矩阵2x3为




```ini
1 2 3
4 5 6
```




行优先存储在内存中为




```shell
1 2 3 4 5 6
```




按照列优先读取为3x2的矩阵，表示如下




```ini
1 4
2 5
3 6
```




转置后为




```ini
1 2 3
4 5 6
```




再之后转换为行优先存储即可

稍微有点绕，读者还请自行理解一下

### 3.2 基本运算操作

在Eigen库中，矩阵的加减乘除操作非常直观。以下是如何在Eigen中执行这些基本矩阵运算的示例：

#### 3.2.1 加法（Addition）

要添加两个矩阵，它们必须具有相同的维度。




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d mat1;
    Eigen::Matrix2d mat2;

    mat1 << 1, 2,
            3, 4;
    mat2 << 5, 6,
            7, 8;

    // 矩阵加法
    Eigen::Matrix2d resultAdd = mat1 + mat2;

    std::cout << "Addition:
" << resultAdd << std::endl;

    return 0;
}
```



#### 3.2.2 减法（Subtraction）

与加法类似，两个矩阵必须具有相同的维度才能进行减法。




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d mat1;
    Eigen::Matrix2d mat2;

    mat1 << 1, 2,
            3, 4;
    mat2 << 5, 6,
            7, 8;

    // 矩阵减法
    Eigen::Matrix2d resultSub = mat1 - mat2;

    std::cout << "Subtraction:
" << resultSub << std::endl;

    return 0;
}
```



#### 3.2.3 元素级别乘除法

在Eigen中，如果你想执行矩阵的元素级别的乘除（即逐元素乘法或逐元素除法），可以使用`array`类型的对象，或者使用特殊的成员函数`cwiseProduct()`和`cwiseQuotient()`。以下是逐元素乘法和除法的示例：

**逐元素乘法（Element-wise Multiplication）**

使用`array`类型：




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d mat1;
    Eigen::Matrix2d mat2;

    mat1 << 1, 2,
            3, 4;
    mat2 << 5, 6,
            7, 8;

    // 逐元素乘法（使用array）
    Eigen::ArrayXXd resultMul = mat1.array() * mat2.array();

    std::cout << "Element-wise Multiplication:
" << resultMul << std::endl;

    return 0;
}
```




**使用`cwiseProduct()`函数：**




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d mat1;
    Eigen::Matrix2d mat2;

    mat1 << 1, 2,
            3, 4;
    mat2 << 5, 6,
            7, 8;

    // 逐元素乘法（使用cwiseProduct）
    Eigen::Matrix2d resultMul = mat1.cwiseProduct(mat2);

    std::cout << "Element-wise Multiplication:
" << resultMul << std::endl;

    return 0;
}
```




**逐元素除法（Element-wise Division），使用`array`类型：**




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d mat1;
    Eigen::Matrix2d mat2;

    mat1 << 1, 2,
            3, 4;
    mat2 << 5, 6,
            7, 8;

    // 逐元素除法（使用array）
    Eigen::ArrayXXd resultDiv = mat1.array() / mat2.array();

    std::cout << "Element-wise Division:
" << resultDiv << std::endl;

    return 0;
}
```




**使用`cwiseQuotient()`函数：**




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d mat1;
    Eigen::Matrix2d mat2;

    mat1 << 1, 2,
            3, 4;
    mat2 << 5, 6,
            7, 8;

    // 逐元素除法（使用cwiseQuotient）
    Eigen::Matrix2d resultDiv = mat1.cwiseQuotient(mat2);

    std::cout << "Element-wise Division:
" << resultDiv << std::endl;

    return 0;
}
```




在这些例子中，`array()`成员函数将矩阵转换为数组类型，允许进行逐元素的算术运算。使用`cwiseProduct()`和`cwiseQuotient()`函数可以在不改变原始矩阵类型的情况下执行逐元素运算。

#### 3.2.4 矩阵乘法（Matrix Multiplication）

两个矩阵进行乘法时，第一个矩阵的列数必须等于第二个矩阵的行数。




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d mat1;
    Eigen::Matrix2d mat2;

    mat1 << 1, 2,
            3, 4;
    mat2 << 5, 6,
            7, 8;

    // 矩阵乘法
    Eigen::Matrix2d resultMul = mat1 * mat2;

    std::cout << "Multiplication:
" << resultMul << std::endl;

    return 0;
}
```



### 3.3 行列归约操作

在NumPy中，可以使用`numpy.sum`和`numpy.mean`函数来按照行或列对数组进行求和和求平均操作。这两个函数都接受一个`axis`参数，该参数指定了沿哪个轴进行操作。

以下是如何按照行或列进行求和和求平均的示例：

按列求和




```python
import numpy as np

# 创建一个2x3的数组
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# 按列求和
sums_cols = np.sum(arr, axis=0)
print(sums_cols)  # 输出: [5 7 9]
```




按行求和




```ini
# 按行求和
sums_rows = np.sum(arr, axis=1)
print(sums_rows)  # 输出: [ 6 15]
```




按列求平均




```ini
# 按列求平均
means_cols = np.mean(arr, axis=0)
print(means_cols)  # 输出: [2.5 3.5 4.5]
```




按行求平均




```shell
# 按行求平均
means_rows = np.mean(arr, axis=1)
print(means_rows)  # 输出: [2. 5.]
```




在这些例子中，`axis=0`表示沿着第一个轴操作，即按列进行操作；`axis=1`表示沿着第二个轴操作，即按行进行操作。如果不指定`axis`参数，`numpy.sum`和`numpy.mean`会计算整个数组的总和和平均值。

---

在Eigen库中，可以使用成员函数`sum()`和`mean()`来分别计算矩阵的行或列的和与平均值。这些操作可以通过指定操作的轴（行或列）来完成。

以下是如何在Eigen中使用这些操作的示例：

按列求和



```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXi mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    // 按列求和
    Eigen::VectorXi colSums = mat.colwise().sum();
    std::cout << "Column sums: " << colSums.transpose() << std::endl;

    return 0;
}
```




按行求和




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXi mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    // 按行求和
    Eigen::VectorXi rowSums = mat.rowwise().sum();
    std::cout << "Row sums: " << rowSums.transpose() << std::endl;

    return 0;
}
```




按列求平均



```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXi mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    // 按列求平均
    Eigen::VectorXd colMeans = mat.colwise().mean();
    std::cout << "Column means: " << colMeans.transpose() << std::endl;

    return 0;
}
```




按行求平均



```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXi mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    // 按行求平均
    Eigen::VectorXd rowMeans = mat.rowwise().mean();
    std::cout << "Row means: " << rowMeans.transpose() << std::endl;

    return 0;
}
```




在这些示例中，`colwise()`和`rowwise()`是Eigen中的操作符，它们允许你在矩阵的列或行上应用函数。`sum()`和`mean()`函数分别计算和与平均值。注意，`mean()`函数返回的是`VectorXd`类型，因为平均值可能是非整数。如果你正在处理整数类型的矩阵，你可能需要将结果转换为适当的类型。

> Note：
> 需要注意的是这里得到的还是一个矩阵，colwise后得到的shape为(1, M)，rowwise后得到的shape为(N, 1)。

在Eigen中，如果你想要计算整个矩阵所有元素的均值（mean）或总和（sum），而不是单独针对行或列，你可以直接使用`mean()`和`sum()`成员函数，而不需要使用`colwise()`或`rowwise()`。

### 3.4 广播操作的转换

Eigen库支持类似于NumPy中的广播（broadcasting）操作，尽管Eigen中的广播概念与NumPy的略有不同。在Eigen中，广播通常指的是将一个较小的数组或标量扩展到一个较大的数组上，以进行逐元素操作。

以下是一些在Eigen中执行广播操作的示例：

1. 将标量广播到矩阵




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    // 将标量广播到矩阵
    double scalar = 10.0;
    Eigen::MatrixXd result = mat.array() + scalar;

    std::cout << "Broadcast scalar to matrix:
" << result << std::endl;

    return 0;
}
```




在这个例子中，标量10.0被广播到矩阵的每个元素上，并与矩阵的每个元素相加。

1. 将小矩阵广播到大矩阵





```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    // 将小矩阵广播到大矩阵
    Eigen::VectorXd vec(3);
    vec << 10, 20, 30;
    Eigen::MatrixXd result = mat.array().colwise() + vec.array();

    std::cout << "Broadcast vector to matrix (column-wise):
" << result << std::endl;

    return 0;
}
```



在这个例子中，向量vec被广播到矩阵mat的每一列上，并与对应列的每个元素相加。

1. 将行向量广播到矩阵





```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    // 将行向量广播到矩阵
    Eigen::RowVectorXd rowVec(3);
    rowVec << 10, 20, 30;
    Eigen::MatrixXd result = mat.array().rowwise() + rowVec.array();

    std::cout << "Broadcast row vector to matrix (row-wise):" << result << std::endl;

    return 0;
}
```



在这个例子中，行向量rowVec被广播到矩阵mat的每一行上，并与对应行的每个元素相加。

在Eigen中，广播通常通过.array()方法将矩阵或向量转换为数组类型，然后使用.colwise()或.rowwise()方法进行逐列或逐行的操作。这允许数组或标量与矩阵的相应部分进行逐元素操作。需要注意的是，广播操作要求被广播的对象的维度必须与目标矩阵的相应维度兼容。

1. 针对不兼容的情况需要自行复制扩充维度

比如在numpy中实现列归一化语句为




```python
a = a / a.sum(axis=0)
```




采用Eigen实现则为：




```cpp
auto conf_res = a.array() / a.colwise().sum().replicate(a.rows(), 1);
```



### 3.5 子矩阵索引

在Eigen中，可以使用几种不同的方法来访问和操作子矩阵。以下是一些常用的方法来索引子矩阵：

**使用`.row()`和`.col()`方法`.row(i)`和`.col(j)`方法分别允许你获取第`i`行和第`j`列。**




```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXi mat(4, 4);
    mat << 1, 2, 3, 4,
           5, 6, 7, 8,
           9, 10, 11, 12,
           13, 14, 15, 16;

    // 获取第2行
    Eigen::VectorXi row = mat.row(1);

    // 获取第3列
    Eigen::VectorXi col = mat.col(2);

    std::cout << "Second row:
" << row << std::endl;
    std::cout << "Third column:
" << col << std::endl;

    return 0;
}
```




**复杂index列表索引**

`‌Eigen::placeholders::all‌`在Eigen库中用于指定矩阵的所有行，当与矩阵的列索引一起使用时，可以用于从矩阵中提取特定的列。在Eigen库中，矩阵的索引可以通过提供列索引和行索引来实现，其中`Eigen::placeholders::all`作为一个占位符，表示选择所有行。




```cpp
#include<iostream>
#include<vector>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;

int main() {
    MatrixXd test_matrix = MatrixXd::Random(4, 6);
    cout << test_matrix << endl << endl;
    // 比如我要依次取出第2列、第5列、第1列
    vector<int> indexs {1, 4, 0}; // 列索引
    MatrixXd index_metrix = test_matrix(Eigen::placeholders::all, indexs); // 使用Eigen::placeholders::all选择所有行，根据indexs选择特定的列
    cout << index_metrix << endl;
    return 0;
}
```



## 四、总结

本文详细总结了在实际工程中，CPU上矩阵运算库在python和C++之间的转换。

实际上，在实际项目中，还会有一些比如矩阵元素排序、查找、求逆、求行列式等运算，一般而言numpy的功能相比于其他库要丰富一些，我们需要知道其内部实现原理，如果其他库没有直接实现，采用最原始的操作内存的方式实现是保底方案。

## 参考文献

* <https://blog.csdn.net/wanzew/article/details/125703765>
* ChatGLM
* <https://blog.csdn.net/sdhdsf132452/article/details/127260531>

