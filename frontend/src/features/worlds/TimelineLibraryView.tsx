import { useEffect, useState, type FormEvent } from "react";

import type { SavePointView, WorldTimelineView } from "../../api/contracts";

interface TimelineLibraryProps {
  timelines: WorldTimelineView[];
  archivedTimelines: WorldTimelineView[];
  busyAction: string | null;
  loadSaves: (worldId: string) => Promise<SavePointView[]>;
  onContinue: (worldId: string) => void;
  onCreateSave: (worldId: string, name: string) => Promise<void>;
  onFork: (worldId: string, saveId: string, name: string) => Promise<void>;
  onArchive: (worldId: string) => Promise<void>;
  onRestore: (worldId: string) => Promise<void>;
}

export function TimelineLibraryView(props: TimelineLibraryProps) {
  const all = [...props.timelines, ...props.archivedTimelines];
  const [selectedId, setSelectedId] = useState(all[0]?.id ?? null);
  const [showArchived, setShowArchived] = useState(false);
  const [saves, setSaves] = useState<SavePointView[]>([]);
  const [saveName, setSaveName] = useState("旅途备忘");
  const [error, setError] = useState<string | null>(null);
  const visible = showArchived ? all : props.timelines;
  const selected = visible.find((timeline) => timeline.id === selectedId) ?? visible[0];

  useEffect(() => {
    if (!selected) {
      setSaves([]);
      return;
    }
    let current = true;
    props.loadSaves(selected.id).then((items) => current && setSaves(items)).catch((caught) => current && setError(caught instanceof Error ? caught.message : "无法读取存档。"));
    return () => { current = false; };
  }, [selected?.id]); // API callback identity is intentionally excluded; selection controls loading.

  async function createSave(event: FormEvent) {
    event.preventDefault();
    if (!selected) return;
    setError(null);
    try {
      await props.onCreateSave(selected.id, saveName.trim());
      setSaves(await props.loadSaves(selected.id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法创建存档。");
    }
  }

  async function fork(save: SavePointView) {
    if (!selected) return;
    const name = `${selected.name} · 分支 ${save.world_version}`;
    setError(null);
    try {
      await props.onFork(selected.id, save.id, name);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法创建分支时间线。");
    }
  }

  async function updateArchiveStatus(operation: () => Promise<void>) {
    setError(null);
    try {
      await operation();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法更新时间线状态。");
    }
  }

  return (
    <section className="content-page timelines-page page-enter">
      <header className="page-header split-header">
        <div><p className="eyebrow">TIMELINE ARCHIVE</p><h1>存档与时间线</h1><p>存档是不可变的书签。从旧存档继续会展开一条新时间线，原本的历史保持不变。</p></div>
        <label className="archive-toggle"><input type="checkbox" checked={showArchived} onChange={(event) => setShowArchived(event.target.checked)} /><span>显示已归档</span></label>
      </header>
      {!visible.length ? <div className="empty-state"><h2>还没有时间线</h2><p>从“开始新旅程”创建第一条 Mistgate 世界线。</p></div> : (
        <div className="timeline-layout">
          <div className="timeline-list" aria-label="时间线列表">
            {visible.map((timeline) => (
              <button key={timeline.id} className={selected?.id === timeline.id ? "timeline-row selected" : "timeline-row"} onClick={() => setSelectedId(timeline.id)}>
                <span className="timeline-glyph">{timeline.forked_from_world_instance_id ? "⑂" : "◈"}</span>
                <span><strong>{timeline.name}</strong><small>{timeline.location_name ?? "尚未进入"} · 世界版本 {timeline.world_version}</small></span>
                <span className={`timeline-status ${timeline.status}`}>{timeline.status === "active" ? "活跃" : "已归档"}</span>
              </button>
            ))}
          </div>
          {selected && <article className="timeline-detail glass-card">
            <div className="detail-heading"><span><small>{selected.world_name}</small><h2>{selected.name}</h2></span><span className="version-chip">v{selected.world_version}</span></div>
            <dl className="timeline-facts"><div><dt>当前位置</dt><dd>{selected.location_name ?? "尚未记录"}</dd></div><div><dt>最近更新</dt><dd>{formatDate(selected.updated_at)}</dd></div><div><dt>来源</dt><dd>{selected.forked_from_world_instance_id ? "旧存档分支" : "新旅程"}</dd></div></dl>
            <div className="detail-actions">
              {selected.status === "active" ? <><button className="primary-button" disabled={!!props.busyAction} onClick={() => props.onContinue(selected.id)}>继续旅程</button><button className="text-button danger" disabled={!!props.busyAction} onClick={() => void updateArchiveStatus(() => props.onArchive(selected.id))}>归档时间线</button></> : <button className="secondary-button" disabled={!!props.busyAction} onClick={() => void updateArchiveStatus(() => props.onRestore(selected.id))}>恢复时间线</button>}
            </div>
            <div className="save-section">
              <div className="section-title"><span><small>IMMUTABLE BOOKMARKS</small><h3>存档书签</h3></span><span>{saves.length} 个</span></div>
              {selected.status === "active" && <form className="quick-save" onSubmit={createSave}><input aria-label="存档名称" value={saveName} onChange={(event) => setSaveName(event.target.value)} maxLength={120} required /><button className="secondary-button" disabled={!!props.busyAction || !saveName.trim()}>创建命名存档</button></form>}
              <div className="save-list">{saves.map((save) => <div className="save-row" key={save.id}><span className="save-node" /><span><strong>{save.name}</strong><small>{save.location_name ?? "未知位置"} · 版本 {save.world_version} · {formatDate(save.created_at)}</small></span><button className="text-button" disabled={!!props.busyAction} onClick={() => void fork(save)}>从此处分支</button></div>)}</div>
            </div>
            {error && <p className="inline-error" role="alert">{error}</p>}
          </article>}
        </div>
      )}
    </section>
  );
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}
