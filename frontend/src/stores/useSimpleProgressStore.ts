/**
 * 简化的进度状态管理 - 基于固定阶段和轮询
 */

import { create } from 'zustand'

export interface SimpleProgress {
  project_id: string
  stage: string
  percent: number
  message: string
  ts: number
  stale_seconds?: number
  stalled_threshold_seconds?: number
  is_stalled?: boolean
  stall_message?: string
}

interface SimpleProgressState {
  // 状态数据
  byId: Record<string, SimpleProgress>
  
  // 轮询控制
  pollingInterval: number | null
  isPolling: boolean
  
  // 操作方法
  upsert: (progress: SimpleProgress) => void
  startPolling: (projectIds: string[], intervalMs?: number) => void
  stopPolling: (projectId?: string) => void
  clearProgress: (projectId: string) => void
  clearAllProgress: () => void
  
  // 获取方法
  getProgress: (projectId: string) => SimpleProgress | null
  getAllProgress: () => Record<string, SimpleProgress>
}

export const useSimpleProgressStore = create<SimpleProgressState>((set, get) => {
  let timer: number | null = null
  const activeProjectIds = new Set<string>()
  let requestInFlight = false

  const stopTimerWhenIdle = () => {
    if (activeProjectIds.size > 0) return
    if (timer !== null) {
      clearInterval(timer)
      timer = null
    }
    set({ isPolling: false, pollingInterval: null })
  }

  const fetchSnapshots = async () => {
    const projectIds = Array.from(activeProjectIds)
    if (projectIds.length === 0 || requestInFlight) return

    requestInFlight = true
    try {
      const queryString = projectIds.map(id => `project_ids=${encodeURIComponent(id)}`).join('&')
      const response = await fetch(`/api/v1/simple-progress/snapshot?${queryString}`)
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`)

      const snapshots: SimpleProgress[] = await response.json()
      snapshots.forEach(snapshot => get().upsert(snapshot))

      snapshots.forEach(snapshot => {
        if (isCompleted(snapshot.stage) || isFailed(snapshot.message)) {
          activeProjectIds.delete(snapshot.project_id)
        }
      })
      stopTimerWhenIdle()
    } catch (error) {
      console.error('轮询进度失败:', error)
    } finally {
      requestInFlight = false
    }
  }

  return {
    // 初始状态
    byId: {},
    pollingInterval: null,
    isPolling: false,

    // 更新或插入进度数据
    upsert: (progress: SimpleProgress) => {
      set((state) => ({
        byId: {
          ...state.byId,
          [progress.project_id]: progress
        }
      }))
    },

    // 开始轮询
    startPolling: (projectIds: string[], intervalMs: number = 5000) => {
      if (projectIds.length === 0) {
        console.warn('没有项目ID，跳过轮询')
        return
      }

      projectIds.forEach(id => activeProjectIds.add(id))
      fetchSnapshots()

      if (timer === null) {
        timer = window.setInterval(fetchSnapshots, intervalMs)
        set({ isPolling: true, pollingInterval: intervalMs })
      }
    },

    // 停止轮询
    stopPolling: (projectId?: string) => {
      if (projectId) activeProjectIds.delete(projectId)
      else activeProjectIds.clear()
      stopTimerWhenIdle()
    },

    // 清除单个项目进度
    clearProgress: (projectId: string) => {
      set((state) => {
        const newById = { ...state.byId }
        delete newById[projectId]
        return { byId: newById }
      })
    },

    // 清除所有进度
    clearAllProgress: () => {
      set({ byId: {} })
    },

    // 获取单个项目进度
    getProgress: (projectId: string) => {
      return get().byId[projectId] || null
    },

    // 获取所有进度
    getAllProgress: () => {
      return get().byId
    }
  }
})

// 阶段显示名称映射
export const STAGE_DISPLAY_NAMES: Record<string, string> = {
  'INGEST': '素材准备',
  'SUBTITLE': '字幕处理',
  'ANALYZE': '内容分析', 
  'HIGHLIGHT': '片段定位',
  'EXPORT': '视频导出',
  'DONE': '处理完成'
}

// 阶段颜色映射
export const STAGE_COLORS: Record<string, string> = {
  'INGEST': '#1890ff',      // 蓝色
  'SUBTITLE': '#52c41a',    // 绿色
  'ANALYZE': '#fa8c16',     // 橙色
  'HIGHLIGHT': '#722ed1',   // 紫色
  'EXPORT': '#eb2f96',      // 粉色
  'DONE': '#13c2c2'         // 青色
}

// 获取阶段显示名称
export const getStageDisplayName = (stage: string): string => {
  return STAGE_DISPLAY_NAMES[stage] || stage
}

// 获取阶段颜色
export const getStageColor = (stage: string): string => {
  return STAGE_COLORS[stage] || '#666666'
}

// 判断是否为完成状态
export const isCompleted = (stage: string): boolean => {
  return stage === 'DONE'
}

// 判断是否为失败状态
export const isFailed = (message: string): boolean => {
  return message.includes('失败') || message.includes('错误') || message.includes('失败')
}
