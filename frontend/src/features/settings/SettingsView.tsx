interface SettingsViewProps {
  locale: string;
  apiConnected: boolean;
  onRetry: () => void;
}

export function SettingsView({ locale, apiConnected, onRetry }: SettingsViewProps) {
  return (
    <section className="content-page page-enter">
      <header className="page-header"><p className="eyebrow">LOCAL PRODUCT SETTINGS</p><h1>设置</h1><p>当前版本为个人本地使用，不包含账号、云同步或公开网络访问。</p></header>
      <div className="settings-grid">
        <article className="settings-card"><span className="settings-icon">◎</span><div><small>Product API</small><h2>{apiConnected ? "本地连接正常" : "本地连接中断"}</h2><p>仅连接 127.0.0.1，前端来源固定为 localhost:5173。</p></div><button className="secondary-button" onClick={onRetry}>重新检查</button></article>
        <article className="settings-card"><span className="settings-icon">文</span><div><small>LANGUAGE</small><h2>简体中文优先</h2><p>当前玩家配置：{locale || "zh-CN"}。完整语言切换将在内容本地化扩展时提供。</p></div></article>
        <article className="settings-card"><span className="settings-icon">⌁</span><div><small>SAVE MODEL</small><h2>分支时间线</h2><p>旧存档不会覆盖当前世界；它会创建保留来源关系的新时间线。</p></div></article>
      </div>
    </section>
  );
}
