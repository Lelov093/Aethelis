import { useState } from "react";

import type { PlaySession } from "./api/contracts";
import { useAethelisProduct } from "./app/useAethelisProduct";
import { HomeView } from "./features/home/HomeView";
import { SettingsView } from "./features/settings/SettingsView";
import { PlayView } from "./features/play/PlayView";
import { AppShell, type ProductView } from "./features/shell/AppShell";
import { NewGameView } from "./features/worlds/NewGameView";
import { TimelineLibraryView } from "./features/worlds/TimelineLibraryView";

export function App() {
  const product = useAethelisProduct();
  const [view, setView] = useState<ProductView>(() => localStorage.getItem("aethelis.activeWorldId") ? "play" : "home");
  const [session, setSession] = useState<PlaySession | null>(null);
  const [activeWorldId, setActiveWorldId] = useState<string | null>(() => localStorage.getItem("aethelis.activeWorldId"));
  const { state } = product;

  async function createTimeline(contentVersionId: string, name: string) {
    const result = await product.createTimeline(contentVersionId, name);
    const activeSession = result.play_session ?? await product.startSession(result.world_instance.id);
    setActiveWorldId(result.world_instance.id);
    localStorage.setItem("aethelis.activeWorldId", result.world_instance.id);
    setSession(activeSession);
    setView("play");
  }

  async function continueTimeline(worldId: string) {
    try {
      const activeSession = await product.startSession(worldId);
      setActiveWorldId(worldId);
      localStorage.setItem("aethelis.activeWorldId", worldId);
      setSession(activeSession);
      setView("play");
    } catch {
      // The product hook exposes the safe API error in the shared connection banner.
    }
  }

  return (
    <AppShell
      activeView={view}
      displayName={state.profile?.display_name ?? "本地旅人"}
      apiConnected={!!state.profile && !state.error}
      onNavigate={setView}
    >
      {state.error && <div className="connection-banner" role="alert"><span><strong>本地世界暂时不可达</strong>{state.error}</span><button onClick={() => void product.refresh()}>重试连接</button></div>}
      {session && view !== "play" && <div className="session-banner" role="status"><span><strong>世界入口已准备</strong>会话 {session.id.slice(0, 8)} · 世界版本 {state.timelines.find((item) => item.id === activeWorldId)?.world_version ?? session.last_observed_world_version}</span><button onClick={() => setView("play")}>进入现场</button></div>}
      {state.loading && !state.profile ? <LoadingView /> : <>
        {view === "home" && <HomeView displayName={state.profile?.display_name ?? "本地旅人"} timelines={state.timelines} busy={!!product.busyAction} onNewGame={() => setView("new-game")} onOpenTimelines={() => setView("timelines")} onContinue={(id) => void continueTimeline(id)} />}
        {view === "new-game" && <NewGameView worlds={state.content} busy={product.busyAction === "create"} onCreate={createTimeline} />}
        {view === "play" && session && activeWorldId && state.profile && <PlayView worldId={activeWorldId} profile={state.profile} initialSession={session} onExit={() => setView("timelines")} onTimelineChanged={product.refresh} />}
        {view === "play" && (!session || !activeWorldId) && <ResumeActiveWorld worldId={activeWorldId} onResume={(id) => void continueTimeline(id)} onExit={() => setView("timelines")} />}
        {view === "timelines" && <TimelineLibraryView timelines={state.timelines} archivedTimelines={state.archivedTimelines} busyAction={product.busyAction} loadSaves={product.listSaves} onContinue={(id) => void continueTimeline(id)} onCreateSave={async (id, name) => { await product.createSave(id, name); }} onFork={async (worldId, saveId, name) => { await product.forkSave(worldId, saveId, name); }} onArchive={async (id) => { await product.archiveTimeline(id); }} onRestore={async (id) => { await product.restoreTimeline(id); }} />}
        {view === "settings" && <SettingsView locale={state.profile?.locale ?? "zh-CN"} apiConnected={!!state.profile && !state.error} onRetry={() => void product.refresh()} />}
      </>}
    </AppShell>
  );
}

function ResumeActiveWorld({ worldId, onResume, onExit }: { worldId: string | null; onResume: (worldId: string) => void; onExit: () => void }) {
  return <section className="play-loading"><p>{worldId ? "检测到上次离开的时间线，可以重新建立本地会话。" : "尚未选择要进入的时间线。"}</p>{worldId && <button className="primary-button" onClick={() => onResume(worldId)}>恢复上次旅程</button>}<button className="text-button" onClick={onExit}>查看时间线</button></section>;
}

function LoadingView() {
  return <div className="loading-view" role="status"><span className="loading-orbit" /><p>正在唤醒本地世界…</p></div>;
}
