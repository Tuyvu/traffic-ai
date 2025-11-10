import { createRouter, createWebHistory } from 'vue-router'

import RuleList from './pages/RuleList.vue'
import RuleCreate from './pages/RuleCreate.vue'
import RuleEdit from './pages/RuleEdit.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/rules', component: RuleList },
    { path: '/rules/create', component: RuleCreate },
    { path: '/rules/:id/edit', component: RuleEdit },
  ]
})
