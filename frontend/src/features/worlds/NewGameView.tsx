import { useState, type FormEvent } from "react";

import type { AvailableWorldContent } from "../../api/contracts";

interface NewGameViewProps {
  worlds: AvailableWorldContent[];
  busy: boolean;
  onCreate: (contentVersionId: string, name: string) => Promise<void>;
}

export function NewGameView({ worlds, busy, onCreate }: NewGameViewProps) {
  const [name, setName] = useState("雾门初航");
  const [selectedId, setSelectedId] = useState(worlds[0]?.content_version_id ?? "");
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      await onCreate(selectedId, name.trim());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法创建时间线。");
    }
  }

  return (
    <section className="content-page page-enter">
      <header className="page-header">
        <p className="eyebrow">NEW TIMELINE</p>
        <h1>开始一条新的时间线</h1>
        <p>新旅程拥有独立的世界历史。启动客户端不会自动创建世界，只有你在此确认后才会写入。</p>
      </header>
      <form className="new-game-layout" onSubmit={submit}>
        <div className="world-selection">
          {worlds.map((world) => (
            <label key={world.content_version_id} className={selectedId === world.content_version_id ? "world-card selected" : "world-card"}>
              <input type="radio" name="world" value={world.content_version_id} checked={selectedId === world.content_version_id} onChange={() => setSelectedId(world.content_version_id)} />
              <span className="world-card-art"><span>霧</span></span>
              <span className="world-card-copy"><small>首个可游玩世界</small><strong>{world.world_name}</strong><span>档案、旧港与分歧的城市。每个角色只知道自己所见。</span></span>
              <span className="selection-mark">✓</span>
            </label>
          ))}
          {!worlds.length && <div className="empty-state"><h2>尚未发现已发布世界</h2><p>请确认本地 API 已完成 Mistgate 内容引导。</p></div>}
        </div>
        <aside className="creation-panel glass-card">
          <span className="card-kicker">时间线信息</span>
          <label className="field-label" htmlFor="timeline-name">时间线名称</label>
          <input id="timeline-name" value={name} onChange={(event) => setName(event.target.value)} maxLength={120} required />
          <div className="creation-note"><strong>不会覆盖其他旅程</strong><p>之后从旧存档出发时，Aethelis 也会创建一条新的分支时间线。</p></div>
          {error && <p className="inline-error" role="alert">{error}</p>}
          <button className="primary-button full-width" disabled={busy || !selectedId || !name.trim()} type="submit">{busy ? "正在建立世界…" : "创建并准备进入"}</button>
        </aside>
      </form>
    </section>
  );
}
