
// import { createContentLoader } from 'vitepress'

// export default createContentLoader('posts/*.md', {
//   includeSrc: false,

//   transform(raw) {
//     return raw
//       .filter(page => page.url !== '/posts/')
//       .map(page => ({
//         title: page.frontmatter.title || '未命名文章',
//         description: page.frontmatter.description || '',
//         date: page.frontmatter.date
//           ? new Date(page.frontmatter.date)
//               .toISOString()
//               .slice(0, 10)
//           : '',
//         url: page.url
//       }))
//       .sort((a, b) => b.date.localeCompare(a.date))
//   }
// })


import { createContentLoader } from 'vitepress'

export default createContentLoader('posts/*.md', {
  includeSrc: false,

  transform(raw) {
    console.log('扫描到的文章：')

    raw.forEach(page => {
      console.log({
        url: page.url,
        title: page.frontmatter.title,
        date: page.frontmatter.date
      })
    })

    return raw
      .filter(page => page.url !== '/posts/')
      .map(page => ({
        title: page.frontmatter.title || '未命名文章',
        description: page.frontmatter.description || '',
        date: page.frontmatter.date
          ? new Date(page.frontmatter.date)
              .toISOString()
              .slice(0, 10)
          : '',
        url: page.url
      }))
      .sort((a, b) => b.date.localeCompare(a.date))
  }
})
