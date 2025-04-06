var divEle = document.createElement("div");

//设置 div 属性，如 id
divEle.setAttribute("class", "social-share");
divEle.setAttribute("style", "text-align:center");
document.getElementsByTagName('article')[0].appendChild(divEle);

var $config = {
    url                                 : decodeURI(window.location.href),
    sites                                :  ['wechat', 'qzone', 'qq', 'weibo', 'douban'], // 启用的站点
    //disabled                        :  ['google', 'facebook', 'twitter'], // 禁用的站点
    wechatQrcodeTitle   :  "微信扫一扫：分享", // 微信二维码提示文字
    wechatQrcodeHelper  : '<p>打开本链接后直接点击分享到朋友圈即可</p>',
};


window.socialShare(divEle, $config);
