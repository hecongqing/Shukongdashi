<template>
  <div id="app">
    <el-container>
      <!-- 头部 -->
      <el-header class="app-header">
        <div class="header-left">
          <img src="./assets/logo.png" alt="Logo" class="logo" />
          <h1 class="title">装备制造故障知识图谱系统</h1>
        </div>
        <div class="header-right">
          <el-button type="primary" @click="toggleTheme">
            <el-icon><Moon v-if="isDark" /><Sunny v-else /></el-icon>
          </el-button>
          <el-dropdown @command="handleCommand">
            <span class="user-info">
              <el-avatar :size="32" src="./assets/avatar.jpg" />
              <span class="username">管理员</span>
            </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="profile">个人资料</el-dropdown-item>
                <el-dropdown-item command="settings">系统设置</el-dropdown-item>
                <el-dropdown-item command="logout" divided>退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>

      <el-container>
        <!-- 侧边栏 -->
        <el-aside width="250px" class="app-aside">
          <el-menu
            :default-active="$route.path"
            class="menu"
            router
            unique-opened
            :collapse="isCollapsed"
          >
            <el-menu-item index="/">
              <el-icon><House /></el-icon>
              <template #title>首页</template>
            </el-menu-item>

            <el-sub-menu index="/diagnosis">
              <template #title>
                <el-icon><Tools /></el-icon>
                <span>故障诊断</span>
              </template>
              <el-menu-item index="/diagnosis/new">新建诊断</el-menu-item>
              <el-menu-item index="/diagnosis/history">诊断历史</el-menu-item>
              <el-menu-item index="/diagnosis/statistics">诊断统计</el-menu-item>
            </el-sub-menu>

            <el-sub-menu index="/knowledge">
              <template #title>
                <el-icon><Connection /></el-icon>
                <span>知识图谱</span>
              </template>
              <el-menu-item index="/knowledge/graph">图谱浏览</el-menu-item>
              <el-menu-item index="/knowledge/entities">实体管理</el-menu-item>
              <el-menu-item index="/knowledge/relations">关系管理</el-menu-item>
            </el-sub-menu>

            <el-sub-menu index="/data">
              <template #title>
                <el-icon><DataBoard /></el-icon>
                <span>数据管理</span>
              </template>
              <el-menu-item index="/data/collection">数据采集</el-menu-item>
              <el-menu-item index="/data/annotation">数据标注</el-menu-item>
              <el-menu-item index="/data/analysis">数据分析</el-menu-item>
            </el-sub-menu>

            <el-sub-menu index="/model">
              <template #title>
                <el-icon><Cpu /></el-icon>
                <span>模型管理</span>
              </template>
              <el-menu-item index="/model/ner">实体抽取</el-menu-item>
              <el-menu-item index="/model/relation">关系抽取</el-menu-item>
              <el-menu-item index="/model/llm">大模型服务</el-menu-item>
            </el-sub-menu>

            <el-menu-item index="/qa">
              <el-icon><ChatLineRound /></el-icon>
              <template #title>智能问答</template>
            </el-menu-item>

            <el-sub-menu index="/system">
              <template #title>
                <el-icon><Setting /></el-icon>
                <span>系统管理</span>
              </template>
              <el-menu-item index="/system/users">用户管理</el-menu-item>
              <el-menu-item index="/system/logs">日志查看</el-menu-item>
              <el-menu-item index="/system/config">系统配置</el-menu-item>
            </el-sub-menu>
          </el-menu>
        </el-aside>

        <!-- 主内容区域 -->
        <el-main class="app-main">
          <div class="breadcrumb-container">
            <el-breadcrumb separator="/">
              <el-breadcrumb-item v-for="item in breadcrumbs" :key="item.path" :to="item.path">
                {{ item.name }}
              </el-breadcrumb-item>
            </el-breadcrumb>
          </div>
          
          <div class="main-content">
            <router-view />
          </div>
        </el-main>
      </el-container>

      <!-- 底部 -->
      <el-footer class="app-footer">
        <div class="footer-content">
          <span>© 2024 装备制造故障知识图谱系统. All rights reserved.</span>
          <div class="footer-links">
            <a href="#" @click="showAbout">关于我们</a>
            <a href="#" @click="showHelp">帮助文档</a>
            <a href="#" @click="showContact">联系我们</a>
          </div>
        </div>
      </el-footer>
    </el-container>

    <!-- 关于对话框 -->
    <el-dialog v-model="aboutDialogVisible" title="关于系统" width="30%">
      <div class="about-content">
        <h3>装备制造故障知识图谱系统</h3>
        <p>版本: 1.0.0</p>
        <p>一个基于知识图谱的装备制造故障诊断专家系统，采用先进的自然语言处理、实体关系抽取、知识图谱构建等技术。</p>
        <div class="tech-stack">
          <h4>技术栈:</h4>
          <ul>
            <li>前端: Vue.js 3 + Element Plus</li>
            <li>后端: Python + FastAPI</li>
            <li>数据库: Neo4j + MySQL + Redis</li>
            <li>AI: transformers + PyTorch</li>
          </ul>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="aboutDialogVisible = false">关闭</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useStore } from 'vuex'
import { ElMessage } from 'element-plus'

export default {
  name: 'App',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useStore()
    
    // 响应式数据
    const isDark = ref(false)
    const isCollapsed = ref(false)
    const aboutDialogVisible = ref(false)
    
    // 计算属性
    const breadcrumbs = computed(() => {
      const matched = route.matched.filter(item => item.name)
      return matched.map(item => ({
        path: item.path,
        name: item.meta?.title || item.name
      }))
    })
    
    // 方法
    const toggleTheme = () => {
      isDark.value = !isDark.value
      document.documentElement.classList.toggle('dark', isDark.value)
    }
    
    const handleCommand = (command) => {
      switch (command) {
        case 'profile':
          router.push('/profile')
          break
        case 'settings':
          router.push('/settings')
          break
        case 'logout':
          ElMessage.success('已退出登录')
          // 这里可以添加实际的退出逻辑
          break
      }
    }
    
    const showAbout = () => {
      aboutDialogVisible.value = true
    }
    
    const showHelp = () => {
      window.open('/help', '_blank')
    }
    
    const showContact = () => {
      ElMessage.info('联系邮箱: admin@example.com')
    }
    
    // 生命周期
    onMounted(() => {
      // 检查本地存储的主题设置
      const savedTheme = localStorage.getItem('theme')
      if (savedTheme === 'dark') {
        isDark.value = true
        document.documentElement.classList.add('dark')
      }
    })
    
    return {
      isDark,
      isCollapsed,
      aboutDialogVisible,
      breadcrumbs,
      toggleTheme,
      handleCommand,
      showAbout,
      showHelp,
      showContact
    }
  }
}
</script>

<style scoped>
.app-header {
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.header-left {
  display: flex;
  align-items: center;
}

.logo {
  height: 40px;
  margin-right: 15px;
}

.title {
  font-size: 20px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 15px;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.username {
  font-size: 14px;
  color: #606266;
}

.app-aside {
  background: #fff;
  border-right: 1px solid #e4e7ed;
  box-shadow: 2px 0 4px rgba(0, 0, 0, 0.05);
}

.menu {
  border-right: none;
  height: 100%;
}

.app-main {
  background: #f5f6fa;
  min-height: calc(100vh - 120px);
}

.breadcrumb-container {
  background: #fff;
  padding: 15px 20px;
  border-radius: 4px;
  margin-bottom: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.main-content {
  padding: 0 20px;
}

.app-footer {
  background: #fff;
  border-top: 1px solid #e4e7ed;
  padding: 15px 20px;
  box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.05);
}

.footer-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
  color: #606266;
}

.footer-links {
  display: flex;
  gap: 20px;
}

.footer-links a {
  color: #409eff;
  text-decoration: none;
  transition: color 0.3s;
}

.footer-links a:hover {
  color: #66b3ff;
}

.about-content {
  text-align: center;
}

.about-content h3 {
  color: #303133;
  margin-bottom: 10px;
}

.about-content p {
  color: #606266;
  line-height: 1.6;
  margin-bottom: 15px;
}

.tech-stack {
  text-align: left;
  margin-top: 20px;
}

.tech-stack h4 {
  color: #303133;
  margin-bottom: 10px;
}

.tech-stack ul {
  color: #606266;
  line-height: 1.8;
}

/* 暗色主题 */
:global(.dark) {
  --el-bg-color: #1a1a1a;
  --el-text-color-primary: #e4e7ed;
  --el-text-color-regular: #cfd3dc;
  --el-border-color: #4c4d4f;
}

:global(.dark) .app-header {
  background: #2d2d2d;
  border-bottom-color: #4c4d4f;
}

:global(.dark) .title {
  color: #e4e7ed;
}

:global(.dark) .app-aside {
  background: #2d2d2d;
  border-right-color: #4c4d4f;
}

:global(.dark) .app-main {
  background: #1a1a1a;
}

:global(.dark) .breadcrumb-container {
  background: #2d2d2d;
}

:global(.dark) .app-footer {
  background: #2d2d2d;
  border-top-color: #4c4d4f;
}
</style>