import { createRouter, createWebHistory } from 'vue-router'
import Text2PerformerViewVue from '@/views/Text2PerformerView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: Text2PerformerViewVue
    }
  ]
})

export default router
