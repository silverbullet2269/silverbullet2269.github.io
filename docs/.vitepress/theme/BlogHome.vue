
<script setup lang="ts">
import { withBase } from 'vitepress'
import { data as posts } from '../../posts/posts.data'
</script>

<template>
  <main class="blog-home">

    <!-- 个人介绍 -->
    <section class="intro">
      <span class="intro-label">
        ✦ PERSONAL TECH BLOG
      </span>

      <h1>
        Hi, I'm
        <span class="highlight">Silverbullet2269.</span> 
      </h1>

      <h2>Learning. Building. Exploring.</h2>

      <p class="intro-desc">
        记录技术学习、开发实践与个人思考。
      </p>
    </section>

    <!-- 内容导航 -->
    <section class="explore">
      <div class="section-heading">
        <h2>Explore</h2>
        <p>学习记录与技术实践</p>
      </div>

      <!-- 文章主入口 -->
      <a :href="withBase('/posts/')" class="article-card">
        <div class="card-top">
          <span class="card-icon">📚</span>
          <span class="card-arrow">↗</span>
        </div>

        <h3>文章目录</h3>
        <p>浏览所有技术文章、学习笔记和实践记录。</p>

        <span class="card-link">
          浏览全部文章 →
        </span>
      </a>

      <!-- 辅助卡片 -->
      <div class="feature-grid">
        <div class="feature-card">
          <span class="feature-icon">📖</span>
          <h3>Notes</h3>
          <p>技术知识、学习笔记与原理总结。</p>
        </div>

        <div class="feature-card">
          <span class="feature-icon">💻</span>
          <h3>Projects</h3>
          <p>项目开发、技术探索与实践经验。</p>
        </div>
      </div>
    </section>

    <!-- 最近文章：自动生成 -->
    <section class="recent-posts">
      <div class="recent-header">
        <h2>Recent Posts</h2>
        <span>Latest updates</span>
      </div>

      <div v-if="posts.length" class="recent-list">
        <a
          v-for="post in posts.slice(0, 3)"
          :key="post.url"
          :href="post.url"
          class="recent-item"
        >
          <div class="recent-content">
            <h3>{{ post.title }}</h3>
            <p>{{ post.description }}</p>
          </div>

          <div class="recent-meta">
            <time>{{ post.date }}</time>
            <span class="recent-arrow">↗</span>
          </div>
        </a>
      </div>

      <p v-else class="empty-posts">
        暂无文章
      </p>
    </section>

  </main>
</template>

<style scoped>
/* ========== 整体布局 ========== */

.blog-home {
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  padding: 50px 0 80px;
}

/* ========== 个人介绍 ========== */

.intro {
  margin-bottom: 50px;
}

.intro-label {
  display: inline-block;
  padding: 6px 14px;
  border-radius: 20px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 1px;
}

.intro h1 {
  margin: 28px 0 18px;
  font-size: clamp(38px, 5.5vw, 60px);
  line-height: 1.2;
  font-weight: 800;
  letter-spacing: -1.5px;
  overflow-wrap: anywhere;
}

.highlight {
  background: linear-gradient(
    120deg,
    #6366f1,
    #06b6d4
  );
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.intro h2 {
  margin: 0;
  padding: 0;
  border: none;
  font-size: clamp(21px, 3vw, 28px);
  font-weight: 600;
  line-height: 1.4;
  color: var(--vp-c-text-1);
}

.intro-desc {
  margin-top: 18px;
  color: var(--vp-c-text-2);
  font-size: 16px;
  line-height: 1.8;
}

/* ========== Explore 标题 ========== */

.section-heading {
  margin-bottom: 20px;
}

.section-heading h2 {
  margin: 0;
  padding: 0;
  border: none;
  font-size: 24px;
  font-weight: 650;
}

.section-heading p {
  margin: 8px 0 0;
  color: var(--vp-c-text-2);
  font-size: 14px;
}

/* ========== 文章主卡片 ========== */

.article-card {
  display: block;
  padding: 22px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  text-decoration: none !important;
  transition:
    border-color 0.2s ease,
    transform 0.2s ease,
    box-shadow 0.2s ease;
}

.article-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
}

.card-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-icon {
  font-size: 24px;
}

.card-arrow {
  color: var(--vp-c-text-3);
  font-size: 22px;
  transition: transform 0.2s ease;
}

.article-card:hover .card-arrow {
  transform: translate(3px, -3px);
  color: var(--vp-c-brand-1);
}

.article-card h3 {
  margin: 14px 0 6px;
  padding: 0;
  border: none;
  font-size: 20px;
  font-weight: 650;
}

.article-card p {
  margin: 0;
  font-size: 14px;
  color: var(--vp-c-text-2);
  line-height: 1.7;
}

.card-link {
  display: inline-block;
  margin-top: 14px;
  color: var(--vp-c-brand-1);
  font-size: 13px;
  font-weight: 600;
}

/* ========== Notes / Projects 卡片 ========== */

.feature-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.feature-card {
  padding: 18px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg);
  transition:
    border-color 0.2s ease,
    transform 0.2s ease;
}

.feature-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
}

.feature-icon {
  font-size: 23px;
}

.feature-card h3 {
  margin: 10px 0 6px;
  padding: 0;
  border: none;
  font-size: 16px;
  font-weight: 600;
}

.feature-card p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 13px;
  line-height: 1.7;
}

/* ========== Recent Posts ========== */

.recent-posts {
  margin-top: 55px;
  padding-top: 28px;
  border-top: 1px solid var(--vp-c-divider);
}

.recent-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.recent-header h2 {
  margin: 0;
  padding: 0;
  border: none;
  font-size: 23px;
  font-weight: 650;
}

.recent-header span {
  color: var(--vp-c-text-3);
  font-size: 12px;
}

.recent-list {
  display: grid;
  gap: 12px;
}

.recent-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  color: var(--vp-c-text-1);
  text-decoration: none !important;
  transition:
    border-color 0.2s ease,
    transform 0.2s ease;
}

.recent-item:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
}

.recent-content {
  min-width: 0;
}

.recent-item h3 {
  margin: 0 0 8px;
  padding: 0;
  border: none;
  font-size: 16px;
  font-weight: 600;
}

.recent-item p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 13px;
  line-height: 1.7;
}

.recent-meta {
  display: flex;
  align-items: center;
  gap: 18px;
  flex-shrink: 0;
}

.recent-meta time {
  color: var(--vp-c-text-3);
  font-size: 12px;
  white-space: nowrap;
}

.recent-arrow {
  color: var(--vp-c-text-3);
  font-size: 18px;
}

.recent-item:hover .recent-arrow {
  color: var(--vp-c-brand-1);
}

.empty-posts {
  color: var(--vp-c-text-3);
  font-size: 14px;
}

/* ========== 手机端 ========== */

@media (max-width: 640px) {
  .blog-home {
    padding: 30px 0 55px;
  }

  .intro {
    margin-bottom: 38px;
  }

  .intro h1 {
    font-size: clamp(32px, 8vw, 42px);
    letter-spacing: -0.8px;
  }

  .intro h2 {
    font-size: 21px;
  }

  .intro-desc {
    font-size: 14px;
  }

  .article-card {
    padding: 20px;
  }

  .feature-grid {
    grid-template-columns: 1fr;
  }

  .recent-item {
    align-items: flex-start;
    flex-direction: column;
    gap: 10px;
  }

  .recent-meta {
    width: 100%;
    justify-content: space-between;
  }
}

@media (prefers-reduced-motion: reduce) {
  .article-card,
  .card-arrow,
  .feature-card,
  .recent-item {
    transition: none;
  }
}
</style>
