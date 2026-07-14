export async function isDesktopMode(): Promise<boolean> {
  if (Boolean((window as any).__TAURI__ || (window as any).__TAURI_INTERNALS__)) {
    return true
  }

  try {
    const response = await fetch('/api/v1/settings/desktop-mode')
    if (!response.ok) {
      return false
    }
    const data = await response.json()
    return Boolean(data?.is_desktop_mode)
  } catch {
    return false
  }
}
