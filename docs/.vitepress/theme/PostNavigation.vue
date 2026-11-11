
<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { data as posts } from '../../posts/posts.data'

const { page } = useData()

const navigation = computed(() => {
  const currentPath = page.value.relativePath

  const currentIndex = posts.findIndex(post => {
    const postPath = decodeURI(post.url)
      .replace(/^\/posts\//, 'posts/')
      .replace(/\.html$/, '')
      .replace(/\/$/, '')

    return postPath === currentPath.replace(/\.md$/, '')
  })

  if (currentIndex === -1) {
    return { prev: null, next: null }
  }

  // 假设 posts 按日期从新到旧排列
  
  return {
    prev: posts[currentIndex - 1] ?? null,
    next: posts[currentIndex + 1] ?? null
  }
})
</script>

<template>
  <nav
    v-if="navigation.prev || navigation.next"
    class="post-navigation"
    aria-label="文章导航"
  >
    <a
      v-if="navigation.prev"
      :href="withBase(navigation.prev.url)"
      class="nav-item"
    >
      <span class="nav-label">← 上一篇</span>
      <strong>{{ navigation.prev.title }}</strong>
    </a>

    <span v-else class="nav-placeholder" />

    <a
      v-if="navigation.next"
      :href="withBase(navigation.next.url)"
      class="nav-item next"
    >
      <span class="nav-label">下一篇 →</span>
      <strong>{{ navigation.next.title }}</strong>
    </a>
  </nav>
</template>

<style scoped>
.post-navigation {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-top: 40px;
  padding-top: 24px;
  border-top: 1px solid var(--vp-c-divider);
}

.nav-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 18px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  text-decoration: none;
  color: var(--vp-c-text-1);
  transition: border-color 0.2s ease;
}

.nav-item:hover {
  border-color: var(--vp-c-brand-1);
}

.nav-label {
  font-size: 12px;
  color: var(--vp-c-text-2);
}

.nav-item strong {
  font-size: 14px;
  line-height: 1.6;
}

.next {
  text-align: right;
}

@media (max-width: 640px) {
  .post-navigation {
    grid-template-columns: 1fr;
  }

  .nav-placeholder {
    display: none;
  }
}
</style>
