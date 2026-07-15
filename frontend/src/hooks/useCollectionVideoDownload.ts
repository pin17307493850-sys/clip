import { useState } from 'react'
import { message } from 'antd'
import { projectApi } from '../services/api'

export const useCollectionVideoDownload = () => {
  const [isGenerating, setIsGenerating] = useState(false)

  const generateAndDownloadCollectionVideo = async (
    projectId: string,
    collectionId: string,
    _collectionTitle: string
  ) => {
    if (isGenerating) return

    setIsGenerating(true)

    try {
      message.info('正在打包合集内的独立切片...')
      await projectApi.downloadCollectionClips(projectId, collectionId)
      message.success('合集切片包下载完成')
    } catch (error) {
      console.error('下载合集切片包失败:', error)
      message.error('下载合集切片包失败')
    } finally {
      setIsGenerating(false)
    }
  }

  return {
    isGenerating,
    generateAndDownloadCollectionVideo
  }
}
