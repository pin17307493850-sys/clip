import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  List,
  Popconfirm,
  Progress,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import {
  CheckCircleFilled,
  DeleteOutlined,
  DownloadOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import {
  speechApi,
  SpeechRecognitionConfig as SpeechConfig,
  WhisperModel,
  WhisperRuntimeStatus,
} from '../services/api'

const { Text, Paragraph } = Typography

interface SpeechRecognitionConfigProps {
  config?: Record<string, unknown>
  onConfigChange?: (config: Record<string, unknown>) => void
}

const accuracyColor: Record<string, string> = {
  '最高': 'green',
  '高': 'green',
  '较好': 'blue',
  '中等': 'gold',
  '较低': 'default',
}

const SpeechRecognitionConfig: React.FC<SpeechRecognitionConfigProps> = () => {
  const [runtime, setRuntime] = useState<WhisperRuntimeStatus | null>(null)
  const [models, setModels] = useState<WhisperModel[]>([])
  const [speechConfig, setSpeechConfig] = useState<SpeechConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [switchingModel, setSwitchingModel] = useState<string | null>(null)
  const timer = useRef<number | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [rt, ms, cfg] = await Promise.all([
        speechApi.getRuntimeStatus(),
        speechApi.getModels(),
        speechApi.getConfig(),
      ])
      setRuntime(rt)
      setModels(Array.isArray(ms) ? ms : [])
      setSpeechConfig(cfg)
    } finally {
      setLoading(false)
    }
  }, [])

  const needsFastPoll = (rt: WhisperRuntimeStatus | null, ms: WhisperModel[]) =>
    rt?.status === 'installing' || ms.some((m) => m.status === 'downloading')

  useEffect(() => {
    refresh()
    return () => {
      if (timer.current) window.clearInterval(timer.current)
    }
  }, [refresh])

  useEffect(() => {
    if (timer.current) window.clearInterval(timer.current)
    const interval = needsFastPoll(runtime, models) ? 2000 : 15000
    timer.current = window.setInterval(refresh, interval)
    return () => {
      if (timer.current) window.clearInterval(timer.current)
    }
  }, [runtime, models, refresh])

  const handleInstall = async () => {
    try {
      const r = await speechApi.installRuntime()
      message.info(r.message || '已开始安装')
      setRuntime((p) => (p ? { ...p, status: 'installing', progress: 5 } : p))
      refresh()
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '安装失败')
    }
  }

  const handleUninstall = async () => {
    try {
      const r = await speechApi.uninstallRuntime()
      message.success(r.message || '已卸载')
      refresh()
    } catch {
      message.error('卸载失败')
    }
  }

  const handleDownload = async (model: string) => {
    try {
      await speechApi.downloadModel(model)
      message.info(`开始下载模型 ${model}`)
      setModels((prev) => prev.map((m) => (m.name === model ? { ...m, status: 'downloading' } : m)))
      refresh()
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '下载失败')
    }
  }

  const handleDelete = async (model: string) => {
    try {
      await speechApi.deleteModel(model)
      message.success(`已删除模型 ${model}`)
      refresh()
    } catch {
      message.error('删除失败')
    }
  }

  const handleUseModel = async (model: string) => {
    if (!speechConfig) return

    setSwitchingModel(model)
    try {
      await speechApi.updateConfig({
        method: speechConfig.method || 'whisper_local',
        whisper_config: {
          ...speechConfig.whisper_config,
          model_name: model,
        },
        enable_fallback: speechConfig.enable_fallback,
        fallback_method: speechConfig.fallback_method,
        output_format: speechConfig.output_format,
      })
      setSpeechConfig((prev) => (
        prev
          ? { ...prev, whisper_config: { ...prev.whisper_config, model_name: model } }
          : prev
      ))
      message.success(`已切换为 ${model}，下一个新任务或重试任务生效`)
      refresh()
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '切换模型失败')
    } finally {
      setSwitchingModel(null)
    }
  }

  const handleUseDevice = async (device: 'auto' | 'cuda' | 'cpu') => {
    if (!speechConfig) return

    const computeType = device === 'cuda' ? 'float16' : device === 'cpu' ? 'int8' : 'auto'
    try {
      await speechApi.updateConfig({
        method: speechConfig.method || 'whisper_local',
        whisper_config: {
          ...speechConfig.whisper_config,
          device,
          compute_type: computeType,
        },
        enable_fallback: speechConfig.enable_fallback,
        fallback_method: speechConfig.fallback_method,
        output_format: speechConfig.output_format,
      })
      setSpeechConfig((prev) => (
        prev
          ? { ...prev, whisper_config: { ...prev.whisper_config, device, compute_type: computeType } }
          : prev
      ))
      message.success(device === 'cuda' ? '已切换为 GPU 加速，新任务生效' : device === 'cpu' ? '已切换为 CPU，新任务生效' : '已切换为自动选择，新任务生效')
      refresh()
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '切换Whisper运行设备失败')
    }
  }

  if (loading) return <Spin />

  const installed = runtime?.status === 'installed'
  const installing = runtime?.status === 'installing'
  const supported = runtime?.platform_supported !== false
  const currentModel = speechConfig?.whisper_config?.model_name
  const currentDevice = speechConfig?.whisper_config?.device || 'auto'

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message="什么时候需要 Whisper？"
        description="导入的视频自带字幕时会直接使用现成字幕。只有当视频没有字幕时，才需要本地 Whisper 自动转写生成字幕。"
      />

      {!supported && (
        <Alert
          type="warning"
          showIcon
          message="当前平台不支持"
          description="当前 Whisper 运行时不可用于此平台。"
        />
      )}

      <Card size="small" title={<Space><ThunderboltOutlined />Whisper 运行时</Space>}>
        {installed && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Space>
              <CheckCircleFilled style={{ color: '#52c41a' }} />
              <Text strong>已安装</Text>
              <Text type="secondary">({(runtime?.packages || []).join(', ')})</Text>
            </Space>
            <Popconfirm
              title="卸载 Whisper 运行时？已下载的模型不会被删除。"
              onConfirm={handleUninstall}
              okText="卸载"
              cancelText="取消"
            >
              <Button danger size="small" icon={<DeleteOutlined />}>卸载运行时</Button>
            </Popconfirm>
          </Space>
        )}

        {installing && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Text>正在安装...{runtime?.message}</Text>
            <Progress percent={runtime?.progress ?? 5} status="active" />
            {runtime?.log_tail && (
              <pre style={{ maxHeight: 120, overflow: 'auto', background: '#1a1a1a', color: '#bbb', padding: 8, fontSize: 11, borderRadius: 4, margin: 0 }}>
                {runtime.log_tail}
              </pre>
            )}
          </Space>
        )}

        {runtime?.status === 'not_installed' && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Paragraph type="secondary" style={{ marginBottom: 8 }}>
              尚未安装。安装会下载 faster-whisper 运行时，完成后再选择并下载一个模型即可使用。
            </Paragraph>
            <Button type="primary" icon={<DownloadOutlined />} onClick={handleInstall} disabled={!supported}>
              安装 Whisper
            </Button>
          </Space>
        )}

        {runtime?.status === 'error' && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Alert type="error" showIcon message="安装出错" description={runtime?.message} />
            <Button icon={<ReloadOutlined />} onClick={handleInstall} disabled={!supported}>重试安装</Button>
          </Space>
        )}
      </Card>

      <Card size="small" title="Whisper 加速">
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            type="info"
            showIcon
            message="推荐选择自动或 GPU"
            description="自动模式会优先检测 CUDA。若 CUDA 运行库不匹配，会自动回退到 CPU。批量导入时字幕识别仍保持串行，避免显存和内存被多个任务同时占满。"
          />
          <Space wrap>
            <Text strong>运行设备</Text>
            <Select
              value={currentDevice}
              style={{ width: 180 }}
              onChange={handleUseDevice}
              options={[
                { value: 'auto', label: '自动（推荐）' },
                { value: 'cuda', label: 'GPU / CUDA' },
                { value: 'cpu', label: 'CPU' },
              ]}
            />
            <Tag color={currentDevice === 'cuda' ? 'green' : currentDevice === 'cpu' ? 'gold' : 'blue'}>
              {currentDevice === 'cuda' ? 'GPU 加速' : currentDevice === 'cpu' ? 'CPU 模式' : '自动选择'}
            </Tag>
          </Space>
        </Space>
      </Card>

      <Card size="small" title="Whisper 模型">
        {installed && currentModel && (
          <Alert
            type="success"
            showIcon
            style={{ marginBottom: 12 }}
            message={`当前使用：${currentModel}`}
            description="切换模型只影响下一个新任务或重试任务，已经开始的字幕识别不会中途切换。"
          />
        )}

        {!installed && (
          <Text type="secondary">请先安装 Whisper 运行时，然后在这里下载模型。</Text>
        )}

        {installed && (
          <List
            dataSource={models}
            renderItem={(m) => {
              const downloaded = m.status === 'downloaded'
              const downloading = m.status === 'downloading'
              const isCurrent = downloaded && m.name === currentModel
              return (
                <List.Item
                  actions={[
                    downloaded ? (
                      <Space>
                        {isCurrent ? (
                          <Button size="small" disabled icon={<CheckCircleFilled />}>当前使用</Button>
                        ) : (
                          <Button
                            size="small"
                            type="primary"
                            loading={switchingModel === m.name}
                            onClick={() => handleUseModel(m.name)}
                          >
                            使用此模型
                          </Button>
                        )}
                        <Popconfirm
                          title={`删除模型 ${m.name}？`}
                          onConfirm={() => handleDelete(m.name)}
                          okText="删除"
                          cancelText="取消"
                        >
                          <Button size="small" danger icon={<DeleteOutlined />} disabled={isCurrent}>删除</Button>
                        </Popconfirm>
                      </Space>
                    ) : downloading ? (
                      <Button size="small" loading disabled>下载中</Button>
                    ) : (
                      <Button size="small" type="primary" icon={<DownloadOutlined />} onClick={() => handleDownload(m.name)}>
                        下载
                      </Button>
                    ),
                  ]}
                >
                  <List.Item.Meta
                    title={
                      <Space wrap>
                        <Text strong>{m.name}</Text>
                        <Text type="secondary">{m.size}</Text>
                        {downloaded && <Tag color="green">已下载</Tag>}
                        {isCurrent && <Tag color="blue">当前使用</Tag>}
                        <Tag color={accuracyColor[m.accuracy] || 'default'}>准确度 {m.accuracy}</Tag>
                        <Tooltip title="速度"><Tag>{m.speed}</Tag></Tooltip>
                      </Space>
                    }
                    description={
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text type="secondary">{m.description}</Text>
                        {downloading && <Progress percent={m.downloadProgress ?? undefined} status="active" />}
                        {m.status === 'error' && m.errorMessage && <Text type="danger">{m.errorMessage}</Text>}
                      </Space>
                    }
                  />
                </List.Item>
              )
            }}
          />
        )}
      </Card>
    </Space>
  )
}

export default SpeechRecognitionConfig
