window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
    },
    "HTML-CSS": { fonts: ["TeX"], linebreaks: { automatic: true } },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    },
    svg: {
        fontCache: 'global'
      }
  };