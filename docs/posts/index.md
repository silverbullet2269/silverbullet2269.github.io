
<script setup>
import { data as posts } from './posts.data.ts'
</script>

# 文章目录

<div class="posts">
  <a
    v-for="post in posts"
    :key="post.url"
    :href="post.url"
    class="post-card"
  >
    <div class="post-title">{{ post.title }}</div>
    <p>{{ post.description }}</p>
    <span class="post-date">{{ post.date }}</span>
  </a>
</div>

<style scoped>
.posts {
  display: grid;
  gap: 16px;
  margin-top: 28px;
}

.post-card {
  display: block;
  padding: 24px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  color: var(--vp-c-text-1);
  text-decoration: none;
  transition: all 0.2s ease;
}

.post-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
}

.post-title {
  font-size: 18px;
  font-weight: 600;
}

.post-card p {
  margin: 10px 0;
  color: var(--vp-c-text-2);
  font-size: 14px;
}

.post-date {
  font-size: 13px;
  color: var(--vp-c-text-3);
}
</style>
