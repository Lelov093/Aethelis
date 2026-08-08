import { useCallback, useEffect, useRef, useState } from "react";

import { aethelisApi, AethelisApiError } from "../../api/client";
import type {
  CommandReceipt,
  ContextualActionView,
  DialogueInteractionView,
  JournalView,
  MapView,
  PlayerProfile,
  PlaySession,
  ResumeSummaryView,
  SceneView,
} from "../../api/contracts";
import { MistgateWorldCanvas } from "../../game/MistgateWorldCanvas";

const TERMINAL_STATUSES = new Set(["completed", "rejected", "cancelled", "failed", "needs_clarification"]);

interface PlayViewProps {
  worldId: string;
  profile: PlayerProfile;
  initialSession: PlaySession;
  onExit: () => void;
  onTimelineChanged: () => Promise<void>;
}

interface ProjectionState {
  scene: SceneView;
  map: MapView;
  journal: JournalView;
  resume: ResumeSummaryView;
}

const SCENE_ASSETS: Record<string, string> = {
  council_square: "/assets/mistgate/council-square-v1.jpg",
  central_archive: "/assets/mistgate/central-archive-v1.jpg",
  market_row: "/assets/mistgate/market-row-v1.jpg",
  workshop_lane: "/assets/mistgate/workshop-lane-v1.jpg",
  old_aqueduct: "/assets/mistgate/old-aqueduct-v1.jpg",
};

const CHARACTER_PORTRAITS: Record<string, string> = {
  mira: "/assets/mistgate/mira-v1.jpg",
  rowan: "/assets/mistgate/rowan-v1.jpg",
};

export function PlayView({ worldId, profile, initialSession, onExit, onTimelineChanged }: PlayViewProps) {
  const [session, setSession] = useState(initialSession);
  const [projection, setProjection] = useState<ProjectionState | null>(null);
  const [activePanel, setActivePanel] = useState<"scene" | "map" | "journal" | "inventory" | "quests" | "actions" | "dialogue" | "narrative" | null>(null);
  const [dialogueTargetId, setDialogueTargetId] = useState<string | null>(null);
  const [dialogueInteractionId, setDialogueInteractionId] = useState<string | null>(null);
  const [conversationMode, setConversationMode] = useState<"idle" | "choices" | "compose">("idle");
  const [conversationHistoryOpen, setConversationHistoryOpen] = useState(false);
  const [nearbyTargetId, setNearbyTargetId] = useState<string | null>(null);
  const [command, setCommand] = useState<CommandReceipt | null>(null);
  const [lastAction, setLastAction] = useState<ContextualActionView | null>(null);
  const [intentText, setIntentText] = useState("");
  const [lastIntentText, setLastIntentText] = useState<string | null>(null);
  const [lastIntentTargetId, setLastIntentTargetId] = useState<string | null>(null);
  const [lastDialogueInteractionId, setLastDialogueInteractionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollGeneration = useRef(0);

  const loadProjection = useCallback(async () => {
    const [scene, map, journal, resume] = await Promise.all([
      aethelisApi.getScene(worldId, profile.id),
      aethelisApi.getMap(worldId, profile.id),
      aethelisApi.getJournal(worldId, profile.id),
      aethelisApi.getResumeSummary(worldId, profile.id),
    ]);
    const next = { scene, map, journal, resume };
    setProjection(next);
    return next;
  }, [profile.id, worldId]);

  useEffect(() => {
    let current = true;
    setLoading(true);
    setError(null);
    loadProjection()
      .catch((caught) => current && setError(messageFor(caught)))
      .finally(() => current && setLoading(false));
    return () => {
      current = false;
      pollGeneration.current += 1;
    };
  }, [loadProjection]);

  async function runAction(
    action: ContextualActionView,
    observedWorldVersion = projection?.scene.world_version,
    interactionId?: string,
  ) {
    if (!projection || commandIsPending(command)) return;
    if (!action.command_required) {
      setActivePanel("scene");
      setCommand(null);
      setLastAction(action);
      return;
    }
    setError(null);
    setLastAction(action);
    setLastIntentText(null);
    setLastIntentTargetId(null);
    setLastDialogueInteractionId(interactionId ?? null);
    const generation = ++pollGeneration.current;
    try {
      const submitted = await aethelisApi.submitContextualCommand(
        worldId,
        {
          player_profile_id: profile.id,
          play_session_id: session.id,
          action_id: action.action_id,
          actor_id: profile.id,
          target_ids: action.target_id ? [action.target_id] : [],
          location_id: action.location_id,
          expected_world_version: observedWorldVersion ?? projection.scene.world_version,
          locale: profile.locale,
          ...(interactionId ? { dialogue_interaction_id: interactionId } : {}),
        },
        commandKey(action.action_id),
      );
      setCommand(submitted);
      await pollCommand(submitted, generation);
    } catch (caught) {
      setError(messageFor(caught));
    }
  }

  async function submitNaturalIntent(
    text: string,
    observedWorldVersion = projection?.scene.world_version,
    focusedTargetId?: string,
    interactionId?: string,
  ) {
    const normalized = text.trim();
    if (!projection || normalized.length < 2 || commandIsPending(command)) return;
    setError(null);
    setLastAction(null);
    setLastIntentText(normalized);
    setLastIntentTargetId(focusedTargetId ?? null);
    setLastDialogueInteractionId(interactionId ?? null);
    const generation = ++pollGeneration.current;
    const allTargetIds = Array.from(new Set([
      ...projection.scene.visible_entities.map((item) => item.id),
      ...projection.scene.visible_resources.map((item) => item.id),
      ...projection.scene.contextual_actions.flatMap((item) => item.target_id ? [item.target_id] : []),
    ]));
    const targetIds = focusedTargetId === "world_narrative"
      ? Array.from(new Set(["world_narrative", ...allTargetIds]))
      : focusedTargetId ? [focusedTargetId] : allTargetIds;
    const targetHints = Object.fromEntries(targetIds.map((id) => {
      const entity = projection.scene.visible_entities.find((item) => item.id === id);
      const resource = projection.scene.visible_resources.find((item) => item.id === id);
      const location = projection.map.locations.find((item) => item.id === id);
      return [id, id === "world_narrative" ? "世界旁白" : entity?.name ?? resource?.name ?? location?.name ?? id];
    }));
    try {
      const submitted = await aethelisApi.submitNaturalLanguageCommand(
        worldId,
        {
          player_profile_id: profile.id,
          play_session_id: session.id,
          text: normalized,
          actor_id: profile.id,
          target_ids: targetIds,
          target_hints: targetHints,
          location_id: projection.scene.location_id,
          expected_world_version: observedWorldVersion ?? projection.scene.world_version,
          locale: profile.locale,
          ...(interactionId ? { dialogue_interaction_id: interactionId } : {}),
        },
        commandKey("natural-intent"),
      );
      setIntentText("");
      setCommand(submitted);
      await pollCommand(submitted, generation);
    } catch (caught) {
      setError(messageFor(caught));
    }
  }

  async function pollCommand(receipt: CommandReceipt, generation: number) {
    let current = receipt;
    while (!TERMINAL_STATUSES.has(current.command.status) && generation === pollGeneration.current) {
      await delay(450);
      current = await aethelisApi.getCommand(worldId, current.command.id);
      if (generation !== pollGeneration.current) return;
      setCommand(current);
    }
    if (generation !== pollGeneration.current) return;
    if (current.command.status === "completed") {
      await Promise.all([loadProjection(), onTimelineChanged()]);
      const resumed = await aethelisApi.startSession(worldId, profile.id);
      setSession(resumed);
    }
  }

  async function cancelCommand() {
    if (!command || !commandIsPending(command)) return;
    pollGeneration.current += 1;
    try {
      setCommand(await aethelisApi.cancelCommand(worldId, command.command.id));
    } catch (caught) {
      setError(messageFor(caught));
    }
  }

  async function retryLastInput() {
    if (!lastAction && !lastIntentText) return;
    const fresh = await loadProjection();
    setCommand(null);
    if (lastAction) {
      await runAction(lastAction, fresh.scene.world_version, lastDialogueInteractionId ?? undefined);
    } else if (lastIntentText) {
      await submitNaturalIntent(
        lastIntentText,
        fresh.scene.world_version,
        lastIntentTargetId ?? undefined,
        lastDialogueInteractionId ?? undefined,
      );
    }
  }

  if (loading) return <section className="play-loading" role="status"><span className="loading-orbit" /><p>正在展开雾门现场……</p></section>;
  if (!projection) return <section className="play-loading"><p>{error ?? "现场投影暂时不可用。"}</p><button className="secondary-button" onClick={onExit}>返回时间线</button></section>;

  const { scene, map, journal, resume } = projection;
  const nearbyEntity = scene.visible_entities.find((item) => item.id === nearbyTargetId);
  const nearbyResource = scene.visible_resources.find((item) => item.id === nearbyTargetId);
  const nearbyName = nearbyEntity?.name ?? nearbyResource?.name ?? null;
  const nearbyActions = nearbyTargetId
    ? scene.contextual_actions.filter((item) => item.target_id === nearbyTargetId)
    : [];
  const dialogueCharacter = dialogueTargetId
    ? scene.visible_entities.find((item) => item.id === dialogueTargetId)
    : undefined;

  function openConversation(target: "world_narrative" | string) {
    setDialogueInteractionId(`dialogue_${crypto.randomUUID()}`);
    setConversationMode("idle");
    setConversationHistoryOpen(false);
    if (target === "world_narrative") {
      setDialogueTargetId(null);
      setActivePanel("narrative");
    } else {
      setDialogueTargetId(target);
      setActivePanel("dialogue");
    }
  }

  function closeConversation() {
    setActivePanel(null);
    setDialogueTargetId(null);
    setDialogueInteractionId(null);
    setConversationMode("idle");
    setConversationHistoryOpen(false);
  }

  function interactWith(targetId: string | null) {
    const entity = targetId ? scene.visible_entities.find((item) => item.id === targetId) : undefined;
    if (entity) {
      openConversation(entity.id);
      return;
    }
    const action = targetId
      ? scene.contextual_actions.find((item) => item.target_id === targetId)
      : undefined;
    if (action) void runAction(action);
    else setActivePanel(targetId ? "scene" : "actions");
  }

  return (
    <section className="play-view immersive-play page-enter" aria-label="雾门游玩现场">
      <MistgateWorldCanvas
        scene={scene}
        backgroundUrl={scene.location_id ? SCENE_ASSETS[scene.location_id] : undefined}
        onNearbyTargetChange={setNearbyTargetId}
        onInteract={interactWith}
      />

      <header className="game-top-hud">
        <button className="game-icon-button" onClick={onExit} aria-label="返回档案">←</button>
        <div className="location-hud"><small>{resume.world_name}</small><strong>{scene.location_name ?? "未知地点"}</strong><span>世界回合 {scene.world_turn} · {scene.elapsed_minutes} 分钟</span></div>
      </header>

      <div className="game-hud-tools" aria-label="游戏工具">
        {(["scene", "map", "inventory", "quests", "journal", "narrative", "actions"] as const).map((panel) => (
          <button
            key={panel}
            className={activePanel === panel ? "active" : ""}
            aria-pressed={activePanel === panel}
            onClick={() => {
              if (panel === "narrative") {
                if (activePanel === "narrative") closeConversation();
                else openConversation("world_narrative");
                return;
              }
              setActivePanel(activePanel === panel ? null : panel);
            }}
          >
            <span aria-hidden="true">{panelGlyph(panel)}</span>{panelLabel(panel)}
          </button>
        ))}
      </div>

      <div className="scene-intro-hud">
        <p className="eyebrow">LIVE WORLD</p>
        <h1>{scene.location_name}</h1>
        <p>{map.locations.find((item) => item.id === scene.location_id)?.summary ?? "雾气正在重组这处地点的轮廓。"}</p>
        <div className={`situation-chip ${journal.situation.phase}`}><span>{situationPhaseLabel(journal.situation.phase)}</span><strong>{journal.situation.completed_steps}/{journal.situation.total_steps} 条关键线索</strong></div>
      </div>

      {journal.outcomes.length > 0 && <div className="world-outcome-hud" role="status"><small>{journal.outcomes[0].outcome_type === "ending" ? "WORLD ENDING" : "WORLD CONTINUES"}</small><strong>{journal.outcomes[0].title}</strong></div>}

      <div className={nearbyName ? "interaction-prompt visible" : "interaction-prompt"}>
        {nearbyName ? <><span><small>附近可互动</small><strong>{nearbyName}</strong></span><button disabled={commandIsPending(command)} onClick={() => interactWith(nearbyTargetId)}><kbd>E</kbd>{nearbyEntity ? "交谈" : nearbyActions[0]?.label ?? "查看"}</button></> : <span><small>探索现场</small><strong>移动靠近人物与物件</strong></span>}
      </div>

      {activePanel && activePanel !== "dialogue" && activePanel !== "narrative" && <article className="game-overlay-panel glass-card" aria-label={`${panelLabel(activePanel)}面板`}>
        <header><div><small>PLAYER VIEW</small><h2>{panelLabel(activePanel)}</h2></div><button className="game-icon-button" onClick={() => setActivePanel(null)} aria-label="关闭面板">×</button></header>
        <div className="game-overlay-scroll">
          {activePanel === "scene" && <ScenePanel scene={scene} onTalk={openConversation} />}
          {activePanel === "map" && <MapPanel map={map} onMove={(location) => {
            const action = scene.contextual_actions.find((item) => item.action_id === "move_to_location" && item.target_id === location.id);
            if (action) void runAction(action);
          }} busy={commandIsPending(command)} />}
          {activePanel === "journal" && <JournalPanel journal={journal} onShowMap={() => setActivePanel("map")} />}
          {activePanel === "inventory" && <InventoryPanel journal={journal} />}
          {activePanel === "quests" && <QuestPanel journal={journal} onShowMap={() => setActivePanel("map")} />}
          {activePanel === "actions" && <div className="action-list immersive-action-list">{scene.contextual_actions.map((action) => <button key={`${action.action_id}:${action.target_id ?? "scene"}`} disabled={commandIsPending(command)} onClick={() => void runAction(action)}><span>{actionGlyph(action.action_id)}</span><strong>{action.label}</strong><small>{action.command_required ? "将改变或检验世界" : "查看现场信息"}</small></button>)}</div>}
        </div>
      </article>}

      {dialogueInteractionId && activePanel === "dialogue" && dialogueCharacter && <ConversationOverlay
        scene={scene}
        targetKind="character"
        targetId={dialogueCharacter.id}
        targetName={dialogueCharacter.name}
        targetSummary={dialogueCharacter.summary}
        interactionId={dialogueInteractionId}
        interactions={journal.dialogue_interactions ?? []}
        options={scene.contextual_actions.filter((item) => item.target_id === dialogueCharacter.id)}
        portraitUrl={CHARACTER_PORTRAITS[dialogueCharacter.id]}
        supportsFreeExpression={scene.supports_free_dialogue}
        recommendedContentVersionId={scene.recommended_content_version_id}
        mode={conversationMode}
        historyOpen={conversationHistoryOpen}
        intentText={intentText}
        busy={commandIsPending(command)}
        command={lastDialogueInteractionId === dialogueInteractionId ? command : null}
        error={lastDialogueInteractionId === dialogueInteractionId ? error : null}
        pendingPlayerText={lastDialogueInteractionId === dialogueInteractionId ? lastIntentText ?? lastAction?.label ?? null : null}
        onModeChange={setConversationMode}
        onHistoryToggle={() => setConversationHistoryOpen(!conversationHistoryOpen)}
        onIntentTextChange={setIntentText}
        onClose={closeConversation}
        onSubmit={(text) => {
          setConversationMode("idle");
          void submitNaturalIntent(text, scene.world_version, dialogueCharacter.id, dialogueInteractionId);
        }}
        onAction={(action) => {
          setConversationMode("idle");
          void runAction(action, scene.world_version, dialogueInteractionId);
        }}
      />}

      {dialogueInteractionId && activePanel === "narrative" && <ConversationOverlay
        scene={scene}
        targetKind="world_narrative"
        targetId={null}
        targetName="世界旁白"
        targetSummary="旁白只描述你当前可见、可听见或已经发现的世界，并将具体尝试交给治理系统。"
        interactionId={dialogueInteractionId}
        interactions={journal.dialogue_interactions ?? []}
        options={scene.contextual_actions.filter((item) => item.action_id === "advance_world")}
        supportsFreeExpression={scene.supports_world_narrative}
        recommendedContentVersionId={scene.recommended_content_version_id}
        mode={conversationMode}
        historyOpen={conversationHistoryOpen}
        intentText={intentText}
        busy={commandIsPending(command)}
        command={lastDialogueInteractionId === dialogueInteractionId ? command : null}
        error={lastDialogueInteractionId === dialogueInteractionId ? error : null}
        pendingPlayerText={lastDialogueInteractionId === dialogueInteractionId ? lastIntentText ?? lastAction?.label ?? null : null}
        onModeChange={setConversationMode}
        onHistoryToggle={() => setConversationHistoryOpen(!conversationHistoryOpen)}
        onIntentTextChange={setIntentText}
        onClose={closeConversation}
        onSubmit={(text) => {
          setConversationMode("idle");
          void submitNaturalIntent(text, scene.world_version, "world_narrative", dialogueInteractionId);
        }}
        onAction={(action) => {
          setConversationMode("idle");
          void runAction(action, scene.world_version, dialogueInteractionId);
        }}
      />}

      {(command || error) && activePanel !== "dialogue" && activePanel !== "narrative" && <div className="command-toast"><CommandPanel command={command} error={error} onCancel={() => void cancelCommand()} onRetry={() => void retryLastInput()} canRetry={!!(lastAction || lastIntentText) && (!!command?.execution.retryable || !!command?.result?.message.toLowerCase().includes("world changed"))} /></div>}
    </section>
  );
}

function ScenePanel({ scene, onTalk }: { scene: SceneView; onTalk?: (characterId: string) => void }) {
  return <div className="panel-content"><div className="fact-grid"><section><h2>在场人物</h2>{scene.visible_entities.length ? scene.visible_entities.map((item) => <div className={CHARACTER_PORTRAITS[item.id] ? "world-fact character-fact" : "world-fact"} key={item.id}>{CHARACTER_PORTRAITS[item.id] && <img src={CHARACTER_PORTRAITS[item.id]} alt={`${item.name}的角色肖像`} loading="lazy" onError={(event) => { event.currentTarget.hidden = true; }} />}<div><strong>{item.name}</strong><p>{item.summary}</p>{onTalk && <button className="text-button" onClick={() => onTalk(item.id)}>与{item.name}交谈</button>}</div></div>) : <p className="muted-copy">附近没有清晰可辨的人物。</p>}</section><section><h2>已发现事物</h2>{scene.visible_resources.length ? scene.visible_resources.map((item) => <div className="world-fact" key={item.id}><strong>{item.name}</strong><p>{item.summary}</p></div>) : <p className="muted-copy">调查现场也许会发现新的线索。</p>}</section></div><section className="public-facts"><h2>公开事实</h2><ul>{scene.public_facts.map((fact) => <li key={fact}>{fact}</li>)}</ul></section></div>;
}

function MapPanel({ map, onMove, busy }: { map: MapView; onMove: (location: MapView["locations"][number]) => void; busy: boolean }) {
  return <div className="panel-content map-list">{map.locations.map((location) => <article key={location.id} className={location.is_current ? "map-location current" : "map-location"}><span className="map-node" /><div><strong>{location.name}</strong><p>{location.summary}</p></div>{location.is_current ? <small>当前位置</small> : <button className="text-button" disabled={busy || !location.is_reachable} onClick={() => onMove(location)}>{location.is_reachable ? "前往" : "尚不可达"}</button>}</article>)}</div>;
}

function JournalPanelBody({ journal, onShowMap }: { journal: JournalView; onShowMap: () => void }) {
  return <div className="panel-content journal-sections"><SituationSummary journal={journal} /><div className="journal-grid"><section><h2>当前目标</h2><ol>{journal.current_objectives.map((item) => <li key={item}>{item}</li>)}</ol></section><section><h2>调查记录</h2>{journal.observations.length ? <ul>{journal.observations.map((item) => <li key={item}>{item}</li>)}</ul> : <p className="muted-copy">尚未记录新的调查发现。</p>}</section></div><section><div className="subsection-heading"><span><small>PLAYER KNOWLEDGE</small><h2>已知与传闻</h2></span><span>{journal.knowledge.length} 条</span></div>{journal.knowledge.length ? <div className="knowledge-list">{journal.knowledge.map((item) => <article className={`knowledge-card ${item.kind}`} key={item.id}><small>{item.kind === "rumor" ? "未验证传闻" : "已确认事实"} · 来源：{item.source_name}</small><p>{item.statement}</p><span>置信度：{confidenceLabel(item.confidence)}</span></article>)}</div> : <p className="muted-copy">与角色交谈后，明确区分的事实和传闻会记录在这里。</p>}</section><section><div className="subsection-heading"><span><small>RELATIONSHIPS</small><h2>人物关系</h2></span><span>{journal.relationships.length} 人</span></div>{journal.relationships.length ? <div className="relationship-list">{journal.relationships.map((item) => <article key={item.character_id}><div><strong>{item.character_name}</strong><p>{item.standing_label}</p></div><span>信任 {item.trust > 0 ? "+" : ""}{item.trust} · 互动 {item.interaction_count}</span></article>)}</div> : <p className="muted-copy">尚未通过有效互动建立人物关系。</p>}</section><section><div className="subsection-heading"><span><small>COMMITMENTS</small><h2>当前承诺</h2></span><span>{journal.commitments.length} 项</span></div>{journal.commitments.length ? <div className="knowledge-list">{journal.commitments.map((item) => <article className="knowledge-card" key={item.id}><small>{commitmentStatusLabel(item.status)} · 对方：{item.counterparty_name}</small><p>{item.description}</p></article>)}</div> : <p className="muted-copy">尚未作出需要后续履行的承诺。</p>}</section><section><div className="subsection-heading"><span><small>OPEN THREADS</small><h2>可追踪的机会</h2></span><button className="text-button" onClick={onShowMap}>查看地图</button></div><div className="opportunity-grid">{journal.opportunities.map((item) => <article className={opportunityClass(item)} key={item.id}><small>{opportunityStateLabel(item)}</small><strong>{item.title}</strong><p>{item.description}</p></article>)}</div></section><section><div className="subsection-heading"><span><small>KNOWN RESOURCES</small><h2>已知资源</h2></span><span>{journal.resources.length} 项</span></div>{journal.resources.length ? <div className="resource-list">{journal.resources.map((item) => <article key={item.id}><span className="resource-glyph">{resourceGlyph(item.kind)}</span><div><strong>{item.name}</strong>{item.is_player_owned && <small>PLAYER INVENTORY</small>}<p>{item.summary}</p><small>{item.custody_label} · 数量 {item.quantity}</small></div></article>)}</div> : <p className="muted-copy">调查地点后，已确认的资源会记录在这里。</p>}</section></div>;
}

function JournalPanel(props: { journal: JournalView; onShowMap: () => void }) {
  const { journal } = props;
  const worldResponses = journal.world_responses ?? [];
  const worldActivities = journal.world_activities ?? [];
  return <div className="journal-response-layout">
    {worldActivities.length > 0 && <section className="panel-content world-response-panel"><div className="subsection-heading"><span><small>LIVING WORLD</small><h2>世界动态</h2></span><span>{worldActivities.length} 条</span></div><div className="knowledge-list">{worldActivities.map((item) => <article className="knowledge-card" key={item.id}><small>回合 {item.turn} · {item.actor_names.join("、")}</small><p>{item.summary}</p></article>)}</div></section>}
    {worldResponses.length > 0 && <section className="panel-content world-response-panel"><div className="subsection-heading"><span><small>WORLD RESPONSES</small><h2>城市回应</h2></span><span>{worldResponses.length} 条</span></div><div className="knowledge-list">{worldResponses.map((item) => <article className={`knowledge-card ${item.response_kind}`} key={item.id}><small>{item.response_kind === "civic_support" ? "协同行动" : "支援撤回"} · 行动者：{item.actor_name}</small><p>{item.summary}</p></article>)}</div></section>}
    <JournalPanelBody {...props} />
  </div>;
}

function InventoryPanel({ journal }: { journal: JournalView }) {
  const items = journal.resources.filter((item) => item.is_player_owned);
  return <div className="panel-content inventory-panel"><p className="muted-copy">这里显示世界快照中确实由你持有的物品；使用、给予或交换仍需可用的治理行动。</p>{items.length ? <div className="inventory-grid">{items.map((item) => <article key={item.id}><span className="resource-glyph">{resourceGlyph(item.kind)}</span><div><strong>{item.name}</strong><p>{item.summary}</p><small>{item.custody_label} · 数量 {item.quantity}</small></div></article>)}</div> : <p className="empty-system-state">背包目前是空的。探索、交换或完成行动后，获得的物品会出现在这里。</p>}</div>;
}

function QuestPanel({ journal, onShowMap }: { journal: JournalView; onShowMap: () => void }) {
  return <div className="panel-content quest-panel"><SituationSummary journal={journal} /><section><small className="eyebrow">CURRENT INTENTIONS</small><h3>当前目标</h3><ol>{journal.current_objectives.map((item) => <li key={item}>{item}</li>)}</ol></section><section><div className="subsection-heading"><span><small>OPEN SITUATIONS</small><h3>可追踪机会</h3></span><button className="text-button" onClick={onShowMap}>查看地图</button></div><div className="quest-list">{journal.opportunities.map((item) => <article className={opportunityClass(item)} key={item.id}><small>{opportunityStateLabel(item)}</small><strong>{item.title}</strong><p>{item.description}</p></article>)}</div></section><p className="muted-copy">任务由当前世界状态推导；零件、钥匙和透镜线索可以按不同顺序推进。</p></div>;
}

function SituationSummary({ journal }: { journal: JournalView }) {
  return <section className={`situation-summary ${journal.situation.phase}`}><div><small>LIVING SITUATION</small><strong>{journal.situation.title}</strong><span>{situationPhaseLabel(journal.situation.phase)} · {journal.situation.completed_steps}/{journal.situation.total_steps}</span></div><p>{journal.situation.summary}</p>{journal.situation.recovery_guidance.length > 0 && <ul>{journal.situation.recovery_guidance.map((item) => <li key={item}>{item}</li>)}</ul>}</section>;
}

function opportunityClass(item: JournalView["opportunities"][number]) {
  return ["opportunity-card", item.is_at_location ? "current" : "", item.is_completed ? "completed" : "", item.is_optional ? "optional" : ""].filter(Boolean).join(" ");
}

function opportunityStateLabel(item: JournalView["opportunities"][number]) {
  if (item.is_completed) return "已推进";
  if (item.is_optional) return `可选线索 · ${item.location_name}`;
  return item.is_at_location ? "就在此处" : item.location_name;
}

function situationPhaseLabel(phase: JournalView["situation"]["phase"]) {
  if (phase === "repaired") return "危机解除";
  if (phase === "contained") return "压力已控制";
  return "调节器持续失稳";
}

function ConversationOverlay({
  scene,
  targetKind,
  targetId,
  targetName,
  targetSummary,
  interactionId,
  interactions,
  options,
  portraitUrl,
  supportsFreeExpression,
  recommendedContentVersionId,
  mode,
  historyOpen,
  intentText,
  busy,
  command,
  error,
  pendingPlayerText,
  onModeChange,
  onHistoryToggle,
  onIntentTextChange,
  onClose,
  onSubmit,
  onAction,
}: {
  scene: SceneView;
  targetKind: "character" | "world_narrative";
  targetId: string | null;
  targetName: string;
  targetSummary: string;
  interactionId: string;
  interactions: DialogueInteractionView[];
  options: ContextualActionView[];
  portraitUrl?: string;
  supportsFreeExpression: boolean;
  recommendedContentVersionId: string | null;
  mode: "idle" | "choices" | "compose";
  historyOpen: boolean;
  intentText: string;
  busy: boolean;
  command: CommandReceipt | null;
  error: string | null;
  pendingPlayerText: string | null;
  onModeChange: (mode: "idle" | "choices" | "compose") => void;
  onHistoryToggle: () => void;
  onIntentTextChange: (value: string) => void;
  onClose: () => void;
  onSubmit: (text: string) => void;
  onAction: (action: ContextualActionView) => void;
}) {
  const current = interactions.find((item) => item.id === interactionId);
  const currentExchanges = current?.exchanges ?? [];
  const latest = currentExchanges[currentExchanges.length - 1];
  const saved = interactions.filter((item) => (
    item.id !== interactionId
    && item.contains_free_expression
    && item.target_kind === targetKind
    && item.target_id === targetId
  ));
  const resultMessage = command && !busy ? command.result?.message ?? null : null;
  const clarificationMessage = command?.command.status === "needs_clarification"
    ? `我还不能确定你的意思。请补充：${(command.execution.parsed_intent?.missing_fields ?? []).map(clarificationLabel).join("、") || "具体行动或目标"}。`
    : null;
  const capabilityMessage = supportsFreeExpression
    ? null
    : recommendedContentVersionId
      ? `这条旧时间线不具备自由对话能力。请返回时间线并使用 ${recommendedContentVersionId} 新建世界。`
      : "这条时间线不具备自由对话能力；仍可使用当前可用的预设选项。";
  const background = scene.location_id && SCENE_ASSETS[scene.location_id]
    ? { backgroundImage: `linear-gradient(180deg,rgba(3,15,20,.18),rgba(3,15,20,.82)),url(${SCENE_ASSETS[scene.location_id]})` }
    : undefined;

  return <section className="conversation-overlay" style={background} aria-label={`与${targetName}对话`}>
    <header className="conversation-global-controls">
      <button onClick={onHistoryToggle} aria-pressed={historyOpen}>历史记录</button>
      <button onClick={onClose}>退出对话</button>
    </header>

    {!historyOpen && portraitUrl && <img className="conversation-portrait" src={portraitUrl} alt={`${targetName}立绘`} />}
    {!historyOpen && <div className="conversation-context"><small>{targetKind === "character" ? "CONVERSATION" : "WORLD NARRATIVE"}</small><h1>{targetName}</h1><p>{targetSummary}</p></div>}

    {historyOpen ? <ConversationHistory current={current} saved={saved} targetName={targetName} /> : <>
      {mode === "compose" && <form className="conversation-composer" onSubmit={(event) => {
        event.preventDefault();
        if (intentText.trim().length >= 2) onSubmit(intentText);
      }}>
        <div><strong>自由表达</strong><button type="button" onClick={() => onModeChange("idle")} aria-label="关闭自由表达">×</button></div>
        <textarea
          autoFocus
          aria-label={targetKind === "character" ? `对${targetName}说` : "向世界旁白表达"}
          value={intentText}
          onChange={(event) => onIntentTextChange(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Escape") onModeChange("idle"); }}
          placeholder={targetKind === "character" ? `对${targetName}说……` : "询问眼前情况，或描述你想尝试的行动……"}
          maxLength={2000}
          disabled={busy}
        />
        <button type="submit" disabled={busy || intentText.trim().length < 2}>发送</button>
      </form>}

      <div className="conversation-mode-controls">
        <div className="conversation-choice-anchor">
          {mode === "choices" && <div className="conversation-choice-list">
            {options.length ? options.map((action) => <button key={`${action.action_id}:${action.target_id ?? "scene"}`} disabled={busy} onClick={() => onAction(action)}>{action.label}</button>) : <p>当前没有可用的预设选项。</p>}
          </div>}
          <button className={mode === "choices" ? "active" : ""} onClick={() => onModeChange(mode === "choices" ? "idle" : "choices")}>预设选项</button>
        </div>
        <button className={mode === "compose" ? "active" : ""} disabled={!supportsFreeExpression} onClick={() => onModeChange(mode === "compose" ? "idle" : "compose")}>自由表达</button>
      </div>

      <div className="conversation-dialogue-box" aria-live="polite">
        <div><small>{latest ? targetName : "对话"}</small><strong>{latest ? "回应" : `正在与${targetName}交谈`}</strong></div>
        {pendingPlayerText && busy ? <><p className="player-line">你：{pendingPlayerText}</p><span className="conversation-waiting">正在等待回应；你仍可查看历史或退出对话。</span></> : latest ? <><p>{latest.response_text}</p>{latest.requested_effect_status !== "none" && <span className="conversation-effect">效果状态：{effectStatusLabel(latest.requested_effect_status)}</span>}</> : clarificationMessage ? <p>{clarificationMessage}</p> : resultMessage ? <p>{translateResult(resultMessage)}</p> : <p>选择一个预设话题，或用自己的话开始交谈。</p>}
        {capabilityMessage && <p className="conversation-capability-note">{capabilityMessage}</p>}
        {error && <p className="inline-error" role="alert">{error}</p>}
      </div>
    </>}
  </section>;
}

function ConversationHistory({ current, saved, targetName }: {
  current: DialogueInteractionView | undefined;
  saved: DialogueInteractionView[];
  targetName: string;
}) {
  return <div className="conversation-history">
    <header><small>DIALOGUE RECORD</small><h2>{targetName} · 交互记录</h2><p>当前交互始终可查看；包含自由表达的交互在退出后保留为一个历史版本。</p></header>
    <section><h3>当前交互</h3>{current?.exchanges.length ? <ConversationTranscript interaction={current} /> : <p className="muted-copy">当前尚未产生对话轮次。</p>}</section>
    <section><h3>历史版本</h3>{saved.length ? [...saved].reverse().map((interaction, index) => <article className="conversation-history-version" key={interaction.id}><h4>交互版本 {saved.length - index}</h4><ConversationTranscript interaction={interaction} /></article>) : <p className="muted-copy">还没有包含自由表达的历史交互。</p>}</section>
  </div>;
}

function ConversationTranscript({ interaction }: { interaction: DialogueInteractionView }) {
  return <ol className="conversation-transcript">{interaction.exchanges.map((exchange) => <li key={exchange.id}><p><strong>你</strong><span>{exchange.input_kind === "preset" ? "预设" : "自由表达"}</span></p><blockquote>{exchange.player_text}</blockquote><p><strong>{interaction.target_name}</strong><span>{effectStatusLabel(exchange.requested_effect_status)}</span></p><blockquote>{exchange.response_text}</blockquote>{exchange.visible_effects.length > 0 && <ul className="conversation-visible-effects">{exchange.visible_effects.map((effect) => <li key={effect}>{effect}</li>)}</ul>}</li>)}</ol>;
}

function CommandPanel({ command, error, onCancel, onRetry, canRetry }: { command: CommandReceipt | null; error: string | null; onCancel: () => void; onRetry: () => void; canRetry: boolean }) {
  const [collapsed, setCollapsed] = useState(false);
  const pending = commandIsPending(command);
  const missing = command?.execution.parsed_intent?.missing_fields ?? [];
  return <aside className={collapsed ? "command-panel glass-card collapsed" : "command-panel glass-card"} aria-live="polite"><header className="response-header"><div><p className="eyebrow">WORLD RESPONSE</p><h2>世界回应</h2></div><button onClick={() => setCollapsed(!collapsed)} aria-label={collapsed ? "展开世界回应" : "折叠世界回应"}>{collapsed ? "展开" : "收起"}</button></header>{collapsed ? command && <div className={`command-status status-${command.command.status}`}><span />{statusLabel(command.command.status)}</div> : <>{!command && !error && <p className="muted-copy">行动结果、世界状态和可见后果会在这里留下记录。</p>}{command && <><div className={`command-status status-${command.command.status}`}><span />{statusLabel(command.command.status)}</div>{command.command.status === "needs_clarification" ? <p className="clarification-message">我还不能确定你的意思。请补充：{missing.map(clarificationLabel).join("、") || "具体行动或目标"}，然后重新表达。</p> : command.result ? <><p className="result-message">{translateResult(command.result.message)}</p>{command.result.consequences.length > 0 && <ul className="consequence-list">{command.result.consequences.map((item) => <li key={item}>{translateConsequence(item)}</li>)}</ul>}</> : command.command.status === "failed" || command.command.status === "rejected" ? <p className="inline-error">{translateIntentError(command.execution.error_code, command.execution.error_message)}</p> : <p className="muted-copy">命令已经持久化，正在等待 World Engine 完成解析与治理。</p>}{pending && <button className="secondary-button full-width" onClick={onCancel}>取消尚未提交的行动</button>}{!pending && command.command.status !== "completed" && command.command.status !== "needs_clarification" && canRetry && <button className="secondary-button full-width" onClick={onRetry}>刷新现场后重试</button>}</>}{error && <p className="inline-error" role="alert">{error}</p>}</>}</aside>;
}

function clarificationLabel(field: string): string {
  return { intent: "你希望采取的行动", action: "具体行动", target: "对象", location: "地点" }[field] ?? field;
}

function translateIntentError(code: string | null, message: string | null): string {
  if (code === "intent_outside_play_boundary") return "这个意图目前无法在世界中安全执行。你可以换一种可实现的说法，或查看行动提示。";
  if (code === "intent_provider_unavailable") return "语言理解服务暂时不可用。你仍可使用附近互动或行动提示，并可稍后重试。";
  if (code === "intent_invalid_output") return "世界没有可靠理解这句话，因此没有产生任何改变。请换一种更明确的说法。";
  return message ?? "这次行动没有改变世界。";
}

function commandIsPending(command: CommandReceipt | null): boolean {
  return !!command && !TERMINAL_STATUSES.has(command.command.status);
}

function commandKey(actionId: string): string {
  return `${actionId}-${Date.now()}-${crypto.randomUUID()}`;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function messageFor(error: unknown): string {
  return error instanceof AethelisApiError ? error.problem.detail : error instanceof Error ? error.message : "本地世界返回了未知错误。";
}

function actionGlyph(actionId: string): string {
  return actionId === "move_to_location" ? "↗" : actionId === "investigate_area" ? "⌕" : actionId === "inspect_resource" ? "◎" : actionId === "ask_character" ? "◇" : actionId === "negotiate_resource" ? "⇄" : "◌";
}

function confidenceLabel(confidence: string): string {
  return { low: "低", medium: "中", high: "高" }[confidence] ?? confidence;
}

function commitmentStatusLabel(status: string): string {
  return { active: "履行中", fulfilled: "已履行", broken: "已违背" }[status] ?? status;
}

function effectStatusLabel(status: string): string {
  return {
    none: "无额外世界效果",
    committed: "效果已提交",
    rejected: "效果被拒绝",
    needs_clarification: "效果需要补充信息",
  }[status] ?? status;
}

function panelLabel(panel: "scene" | "map" | "journal" | "inventory" | "quests" | "actions" | "dialogue" | "narrative"): string {
  return { scene: "现场", map: "地图", journal: "日志", inventory: "背包", quests: "任务", actions: "行动", dialogue: "对话", narrative: "旁白" }[panel];
}

function panelGlyph(panel: "scene" | "map" | "journal" | "inventory" | "quests" | "actions" | "narrative"): string {
  return { scene: "◎", map: "◇", journal: "▤", inventory: "▣", quests: "✓", actions: "✦", narrative: "◈" }[panel];
}

function resourceGlyph(kind: string): string {
  return { key_item: "钥", material: "材", information: "讯", service: "助" }[kind] ?? "物";
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = { submitted: "已提交", interpreting: "正在解析", ready_for_governance: "等待治理", verifying: "正在验证", completed: "世界已更新", rejected: "行动被拒绝", cancelled: "行动已取消", failed: "执行失败", needs_clarification: "需要补充信息" };
  return labels[status] ?? status;
}

function translateResult(message: string): string {
  if (message.startsWith("You moved to ")) return `你已抵达${message.slice(13)}`;
  if (message.startsWith("You arrived at ")) return `你已抵达${message.slice(15).replace(/\.$/, "")}。`;
  if (message.startsWith("You discovered ")) return `你发现了${message.slice(15).replace(/\.$/, "")}。`;
  if (message === "Nothing new could be discovered here.") return "这里暂时没有新的发现。";
  if (message.includes("world changed")) return "在行动完成前，世界已经发生变化。请刷新现场后重试。";
  return message;
}

function translateConsequence(value: string): string {
  if (value.startsWith("Moved to location: ")) return `位置改变：${value.slice(19)}`;
  if (value.startsWith("Moved to: ")) return `位置改变：${value.slice(10)}`;
  if (value.startsWith("Discovered resource: ")) return `发现资源：${value.slice(21)}`;
  if (value.startsWith("Knowledge recorded: ")) return `知识已记录：${value.slice(20)}`;
  if (value.startsWith("Relationship changed: ")) return `关系变化：${value.slice(22)}`;
  if (value.startsWith("Resource acquired: ")) return `获得资源：${value.slice(19)}`;
  if (value.startsWith("Commitment recorded: ")) return `承诺已记录：${value.slice(21)}`;
  if (value.startsWith("Local stock remaining: ")) return `当地库存剩余：${value.slice(23)}`;
  if (value.startsWith("Resource consumed: ")) return `消耗资源：${value.slice(19)}`;
  if (value.startsWith("Commitment fulfilled: ")) return `承诺已履行：${value.slice(22)}`;
  if (value.startsWith("Commitment broken: ")) return `承诺已违背：${value.slice(19)}`;
  if (value.startsWith("Commitment remains broken: ")) return `承诺仍处于违背状态：${value.slice(27)}`;
  if (value.startsWith("Repair progress recorded: ")) return `修复进展已记录：${value.slice(26)}`;
  if (value.startsWith("Outcome reached: ")) return `世界状态已达成：${value.slice(17)}`;
  if (value === "Held repair materials remain in player custody.") return "维修材料仍由你持有。";
  return value;
}
