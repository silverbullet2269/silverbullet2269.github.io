// import { defineConfig } from 'vitepress'

// // https://vitepress.dev/reference/site-config
// export default defineConfig({
//   title: "Silverbullet2269's Technical Blog",
//   description: "Technical Blog",
//   themeConfig: {
//     // https://vitepress.dev/reference/default-theme-config
//     nav: [
//       { text: 'Home', link: '/' },
//       { text: 'Examples', link: '/markdown-examples' }
//     ],

//     sidebar: [
//       {
//         text: 'Examples',
//         items: [
//           { text: 'Markdown Examples', link: '/markdown-examples' },
//           { text: 'Runtime API Examples', link: '/api-examples' }
//         ]
//       }
//     ],

//     socialLinks: [
//       { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
//     ]
//   }
// })





import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Silverbullet2269's Technical Blog",
  description: 'My Technical Blog',
  lang: 'zh-CN',

  base: '/',
  cleanUrls: true,
  lastUpdated: true,

  head: [
    ['meta', {
      name: 'author',
      content: 'Silverbullet2269'
    }],
    ['meta', {
      name: 'keywords',
      content: 'Technical Blog, Learning Notes'
    }],
    ['link', {
      rel: 'icon',
      href: '/favicon.svg'
    }]
  ],

  markdown: {
    math: true
  },

  themeConfig: {
    siteTitle: "Silverbullet2269's Technical Blog",

    // 顶部导航
    nav: [
      { text: '首页', link: '/' },
      { text: '文章', link: '/posts/' },
      { text: '关于我', link: '/about' }
    ],

    // 暂时不设置手动侧边栏
    // 后续可以通过自动生成目录的方式管理文章
    sidebar: {},

    search: {
      provider: 'local'
    },

    socialLinks: [
      {
        icon: 'github',
        link: 'https://github.com/Silverbullet2269'
      }
    ],

    outline: {
      level: [2, 3],
      label: '本文目录'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    lastUpdated: {
      text: '最后更新于'
    },

    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '文章目录',
    darkModeSwitchLabel: '主题切换',

    footer: {
      message: '基于 VitePress 构建',
      copyright: 'Copyright © 2026 Silverbullet2269'
    }
  }
})

