
import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'

import BlogHome from './BlogHome.vue'
import PostNavigation from './PostNavigation.vue'
import './custom.css'

export default {
  extends: DefaultTheme,

  Layout() {
    return h(DefaultTheme.Layout, null, {
      'doc-after': () => h(PostNavigation)
    })
  },

  
  enhanceApp({ app }) {
    app.component('BlogHome', BlogHome)
  }
}
