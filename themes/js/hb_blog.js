/* 海滨的Blog —— 站点增强（无框架、无首页外部依赖）
 * 1. 阅读进度条与上下浮动按钮
 * 2. 首页“开始阅读”打开导航目录
 * 3. 仅公式页加载 MathJax（BootCDN）
 * 4. 仅文章页加载 social-share（BootCDN）
 */
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  function loadStyle(href) {
    if (document.querySelector('link[href="' + href + '"]')) return;
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  }

  function loadScript(src, done) {
    var existing = document.querySelector('script[src="' + src + '"]');
    if (existing) {
      if (done) existing.addEventListener("load", done, { once: true });
      return;
    }
    var script = document.createElement("script");
    script.src = src;
    script.async = true;
    if (done) script.addEventListener("load", done, { once: true });
    document.head.appendChild(script);
  }

  function loadMathJaxWhenNeeded() {
    if (!document.querySelector(".arithmatex")) return;
    window.MathJax = {
      tex: {
        inlineMath: [["$", "$"], ["\\(", "\\)"]],
        displayMath: [["$$", "$$"], ["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
      },
      "HTML-CSS": { fonts: ["TeX"], linebreaks: { automatic: true } },
      options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex|md-ellipsis"
      },
      svg: { fontCache: "global" }
    };
    loadScript("https://lib.baomitu.com/mathjax/3.2.2/es5/tex-mml-chtml.js");
  }

  function setupLazyShare() {
    if (!document.querySelector(".hb-rt--article")) return;
    var article = document.querySelector("article");
    if (!article) return;

    var marker = document.createElement("div");
    marker.className = "hb-share-lazy";
    marker.setAttribute("aria-label", "分享文章");
    var comments = article.querySelector(".hb-comments");
    if (comments) article.insertBefore(marker, comments);
    else article.appendChild(marker);

    var loaded = false;
    var observer;
    function loadShare() {
      if (loaded) return;
      loaded = true;
      if (observer) observer.disconnect();
      loadStyle("https://cdn.bootcdn.net/ajax/libs/social-share.js/1.0.16/css/share.min.css");
      loadScript("https://cdn.bootcdn.net/ajax/libs/social-share.js/1.0.16/js/social-share.min.js", function () {
        if (typeof window.socialShare !== "function") return;
        marker.className = "social-share";
        window.socialShare(marker, {
          url: decodeURI(window.location.href),
          sites: ["wechat", "qzone", "qq", "weibo", "douban"],
          wechatQrcodeTitle: "微信扫一扫：分享",
          wechatQrcodeHelper: "<p>打开本链接后直接点击分享到朋友圈即可</p>"
        });
      });
    }

    if ("IntersectionObserver" in window) {
      observer = new IntersectionObserver(function (entries) {
        if (entries.some(function (entry) { return entry.isIntersecting; })) loadShare();
      }, { rootMargin: "700px 0px" });
      observer.observe(marker);
    } else {
      loadShare();
    }
  }

  function installPrintToc() {
    if (document.querySelector(".hb-home")) return;
    var article = document.querySelector("article.md-content__inner");
    var source = document.querySelector(".md-sidebar--secondary .md-nav--secondary");
    var heading = article && article.querySelector("h1");
    var sourceList = source && source.querySelector(".md-nav__list");
    if (!article || !heading || !sourceList || article.querySelector(".hb-print-toc")) return;

    var printToc = document.createElement("nav");
    printToc.className = "hb-print-toc";
    printToc.setAttribute("aria-label", "打印目录");
    var title = document.createElement("div");
    title.className = "hb-print-toc__title";
    title.textContent = "目录";
    var list = sourceList.cloneNode(true);
    list.removeAttribute("data-md-component");
    Array.prototype.forEach.call(list.querySelectorAll(".md-nav__link--active"), function (link) {
      link.classList.remove("md-nav__link--active");
    });
    printToc.appendChild(title);
    printToc.appendChild(list);

    var anchor = heading.nextElementSibling;
    if (!anchor || !anchor.classList.contains("hb-rt--article")) anchor = heading;
    anchor.insertAdjacentElement("afterend", printToc);
  }

  ready(function () {
    loadMathJaxWhenNeeded();
    setupLazyShare();
    installPrintToc();

    var bar = document.createElement("div");
    bar.className = "hb-progress";
    document.body.appendChild(bar);

    var fab = document.createElement("div");
    fab.className = "hb-fab";
    var btnUp = document.createElement("button");
    btnUp.className = "hb-fab-btn hb-fab-up";
    btnUp.title = "回到顶部";
    btnUp.setAttribute("aria-label", "回到顶部");
    btnUp.innerHTML = "<svg viewBox='0 0 24 24' width='14' height='14'><path fill='currentColor' d='M12 5l7 7-1.4 1.4L13 8.8V20h-2V8.8l-4.6 4.6L5 12z'/></svg>";
    var btnDown = document.createElement("button");
    btnDown.className = "hb-fab-btn hb-fab-down";
    btnDown.title = "直达底部";
    btnDown.setAttribute("aria-label", "直达底部");
    btnDown.innerHTML = "<svg viewBox='0 0 24 24' width='14' height='14'><path fill='currentColor' d='M12 19l-7-7 1.4-1.4L11 15.2V4h2v11.2l4.6-4.6L19 12z'/></svg>";
    fab.appendChild(btnUp);
    fab.appendChild(btnDown);
    document.body.appendChild(fab);

    btnUp.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    btnDown.addEventListener("click", function () {
      window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
    });

    var ticking = false;
    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        var doc = document.documentElement;
        var height = doc.scrollHeight - window.innerHeight;
        bar.style.width = (height > 0 ? (window.scrollY / height) * 100 : 0) + "%";
        btnUp.classList.toggle("hb-show", window.scrollY > 300);
        btnDown.classList.toggle("hb-show", height > 600 && window.scrollY < height - 300);
        ticking = false;
      });
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    var desktopNav = window.matchMedia("(min-width: 76.25em)");
    var menu = document.querySelector('.md-header__button[for="__drawer"]');
    if (menu) {
      menu.addEventListener("click", function (event) {
        if (!desktopNav.matches) return;
        event.preventDefault();
        if (document.querySelector(".hb-home")) {
          document.body.classList.toggle("hb-nav-open");
        } else {
          document.body.classList.toggle("hb-sidebar-collapsed");
        }
      });
    }

    var start = document.getElementById("hb-start");
    if (start) {
      start.addEventListener("click", function (event) {
        event.preventDefault();
        if (desktopNav.matches) {
          document.body.classList.add("hb-nav-open");
        } else {
          var drawer = document.getElementById("__drawer");
          if (drawer) drawer.checked = true;
        }
      });
    }
  });
})();
