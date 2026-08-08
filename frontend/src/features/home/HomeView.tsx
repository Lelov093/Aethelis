import type { WorldTimelineView } from "../../api/contracts";
import { MistgateAmbientCanvas } from "../../game/MistgateAmbientCanvas";

interface HomeViewProps {
  displayName: string;
  timelines: WorldTimelineView[];
  busy: boolean;
  onNewGame: () => void;
  onOpenTimelines: () => void;
  onContinue: (worldId: string) => void;
}

export function HomeView({ displayName, timelines, busy, onNewGame, onOpenTimelines, onContinue }: HomeViewProps) {
  const recent = timelines[0];
  return (
    <section className="home-view page-enter">
      <div className="hero-panel">
        <MistgateAmbientCanvas />
        <div className="hero-copy">
          <p className="eyebrow">MISTGATE ARCHIVE CITY · LOCAL WORLD</p>
          <h1>雾起时，<br />档案会记住你的选择。</h1>
          <p className="hero-summary">{displayName}，欢迎回来。这里的道路、传闻与人物关系会沿着每条时间线继续生长。</p>
          <div className="hero-actions">
            {recent ? (
              <button className="primary-button" disabled={busy} onClick={() => onContinue(recent.id)}>
                继续最近旅程
              </button>
            ) : (
              <button className="primary-button" onClick={onNewGame}>进入雾门</button>
            )}
            <button className="ghost-button" onClick={onOpenTimelines}>查看时间线</button>
          </div>
        </div>
        {recent && (
          <article className="recent-journey glass-card">
            <span className="card-kicker">最近的时间线</span>
            <h2>{recent.name}</h2>
            <p>{recent.location_name ?? "位置尚未记录"}</p>
            <div className="journey-meta"><span>世界版本 {recent.world_version}</span><span>{formatDate(recent.updated_at)}</span></div>
          </article>
        )}
      </div>
      <div className="home-lower-grid">
        <article className="manifesto-card">
          <span className="card-kicker">世界规则</span>
          <h2>你可以影响世界，但不能绕过世界。</h2>
          <p>行动会经过验证、提交并形成可追溯的后果。存档保留已发生的历史；从旧存档出发，会生成新的时间线。</p>
        </article>
        <button className="new-journey-card" onClick={onNewGame}>
          <span className="plus-mark">＋</span><span><strong>开始新旅程</strong><small>创建独立的 Mistgate 时间线</small></span>
        </button>
      </div>
    </section>
  );
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}
