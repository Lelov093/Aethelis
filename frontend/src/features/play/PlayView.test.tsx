import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { aethelisApi } from "../../api/client";
import type { CommandReceipt, JournalView, MapView, PlayerProfile, PlaySession, ResumeSummaryView, SceneView } from "../../api/contracts";
import { PlayView } from "./PlayView";

const profile: PlayerProfile = {
  id: "profile_1",
  principal_id: "principal_1",
  display_name: "雾门旅人",
  locale: "zh-CN",
  accessibility_preferences: {},
  created_at: "2026-08-02T10:00:00Z",
  updated_at: "2026-08-02T10:00:00Z",
};

const session: PlaySession = {
  id: "session_1",
  world_instance_id: "world_1",
  player_profile_id: profile.id,
  status: "active",
  entry_world_version: 0,
  last_observed_world_version: 0,
};

const scene: SceneView = {
  world_instance_id: "world_1",
  world_version: 0,
  world_turn: 0,
  elapsed_minutes: 0,
  location_id: "council_square",
  location_name: "议会广场",
  visible_entities: [{ id: "rowan", name: "罗文·凯斯特", summary: "守卫队长。", status: "active" }],
  visible_resources: [],
  public_facts: ["调节器需要维修。"],
  contextual_actions: [{
    action_id: "move_to_location",
    label: "前往：中央档案馆",
    location_id: "council_square",
    target_id: "central_archive",
    command_required: true,
  }, {
    action_id: "advance_world",
    label: "等待世界推进",
    location_id: "council_square",
    target_id: null,
    command_required: true,
  }],
  content_version_id: "mistgate_product_v1_10_0",
  supports_free_dialogue: true,
  supports_world_narrative: true,
  recommended_content_version_id: null,
};

const map: MapView = {
  world_instance_id: "world_1",
  world_version: 0,
  current_location_id: "council_square",
  locations: [{ id: "council_square", name: "议会广场", summary: "入口。", accessibility_label: "入口", is_current: true, is_reachable: false }],
};

const journal: JournalView = { world_instance_id: "world_1", world_version: 0, entries: [], confirmed_facts: [], observations: [], current_objectives: ["选择第一条线索。"], resources: [{ id: "inventory_stabilizer_parts", name: "稳定器零件", summary: "一组紧缺的维修材料。", kind: "material", quantity: 1, custody_label: "由你持有", source_resource_id: "stabilizer_parts", is_player_owned: true }, { id: "repair_record", name: "维修记录", summary: "一份公开记录。", kind: "information", quantity: 1, custody_label: "位于中央档案馆", source_resource_id: "repair_record", is_player_owned: false }], opportunities: [{ id: "archive_records", title: "查阅旧维修记录", description: "档案也许能说明维修条件。", location_id: "central_archive", location_name: "中央档案馆", action_id: "move_to_location", target_id: "central_archive", is_at_location: false, is_completed: false, is_optional: true }], situation: { phase: "contained", title: "城市暂时守住了", summary: "危机受到控制，世界可以继续发展。", completed_steps: 2, total_steps: 4, recovery_guidance: ["前往工坊巷调查校准钥匙。"] }, knowledge: [{ id: "knowledge_key_rumor", kind: "rumor", statement: "有人说钥匙曾经过集市，但尚未验证。", source_name: "娜拉·维伊", confidence: "low" }], relationships: [{ character_id: "nara", character_name: "娜拉·维伊", trust: 1, standing_label: "初步信任", interaction_count: 1 }], commitments: [{ id: "commitment_repair", counterparty_name: "赛尔卡", description: "提供调节器维修已经推进的证据。", status: "active" }], outcomes: [{ id: "outcome_city_holds", outcome_type: "stable_continuation", title: "城市暂时守住了", description: "危机受到控制，世界可以继续发展。" }] };
const resume: ResumeSummaryView = { world_instance_id: "world_1", world_version: 0, world_name: "雾门档案城", location_name: "议会广场", last_save_reason: "initial", visible_resource_count: 0, resumable_session_id: session.id };

function receipt(status: CommandReceipt["command"]["status"]): CommandReceipt {
  return {
    command: { id: "command_1", action_id: "move_to_location", status, cancellation_requested: false, updated_at: "2026-08-02T10:00:00Z" },
    execution: { attempt_count: 0, max_attempts: 3, error_code: null, error_message: null, retryable: false },
    status_url: "/commands/command_1",
    result: status === "rejected" || status === "cancelled" ? { command_id: "command_1", status, message: "The world changed before this action could commit. Try again.", source_world_version: 0, resulting_world_version: null, snapshot_id: null, consequences: [], available_actions: ["return_to_scene"], created_at: "2026-08-02T10:00:00Z" } : null,
  };
}

function setupProjectionMocks() {
  vi.spyOn(aethelisApi, "getScene").mockResolvedValue(scene);
  vi.spyOn(aethelisApi, "getMap").mockResolvedValue(map);
  vi.spyOn(aethelisApi, "getJournal").mockResolvedValue(journal);
  vi.spyOn(aethelisApi, "getResumeSummary").mockResolvedValue(resume);
  vi.spyOn(aethelisApi, "startSession").mockResolvedValue(session);
}

async function openRowanDialogue() {
  fireEvent.click(await screen.findByRole("button", { name: "现场" }));
  fireEvent.click(screen.getByRole("button", { name: "与罗文·凯斯特交谈" }));
}

describe("PlayView command recovery", () => {
  afterEach(() => vi.restoreAllMocks());

  it("renders a full game surface and keeps character detail in a compact overlay", async () => {
    setupProjectionMocks();
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    expect(await screen.findByRole("application", { name: /议会广场 可移动游戏场景/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "现场" }));
    expect(screen.getByRole("img", { name: "罗文·凯斯特的角色肖像" })).toHaveAttribute("src", "/assets/mistgate/rowan-v1.jpg");
  });

  it("shows player-safe opportunities and discovered resource custody in the journal", async () => {
    setupProjectionMocks();
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    fireEvent.click(await screen.findByRole("button", { name: "日志" }));
    expect(screen.getByText("查阅旧维修记录")).toBeInTheDocument();
    expect(screen.getByText("位于中央档案馆 · 数量 1")).toBeInTheDocument();
    expect(screen.getByText("未验证传闻 · 来源：娜拉·维伊")).toBeInTheDocument();
    expect(screen.getByText("信任 +1 · 互动 1")).toBeInTheDocument();
    expect(screen.getByText("提供调节器维修已经推进的证据。")).toBeInTheDocument();
    expect(screen.getByText("履行中 · 对方：赛尔卡")).toBeInTheDocument();
    expect(screen.getByText("由你持有 · 数量 1")).toBeInTheDocument();
    expect(screen.getAllByText("城市暂时守住了")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "查看地图" }));
    expect(screen.getByRole("button", { name: "地图" })).toHaveAttribute("aria-pressed", "true");
  });

  it("keeps a committed autonomous city response visible in the journal", async () => {
    setupProjectionMocks();
    vi.mocked(aethelisApi.getJournal).mockResolvedValue({
      ...journal,
      world_responses: [{
        id: "response_selka",
        actor_name: "Selka Orin",
        response_kind: "civic_support",
        summary: "The guild has committed follow-up maintenance supplies.",
      }],
    });
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    fireEvent.click(await screen.findByRole("button", { name: "日志" }));
    expect(screen.getByText("城市回应")).toBeInTheDocument();
    expect(screen.getByText("The guild has committed follow-up maintenance supplies.")).toBeInTheDocument();
  });

  it("shows persisted living-world activity and offers bounded narrative input", async () => {
    setupProjectionMocks();
    vi.mocked(aethelisApi.getJournal).mockResolvedValue({
      ...journal,
      world_activities: [{
        id: "activity_1",
        turn: 1,
        actor_names: ["罗文·凯斯特", "塔伦·索尔"],
        activity_kind: "knowledge_propagation",
        summary: "罗文把玩家提供的说法转告给塔伦；消息仍未验证。",
      }],
    });
    const submit = vi.spyOn(aethelisApi, "submitNaturalLanguageCommand").mockResolvedValue(receipt("needs_clarification"));
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    fireEvent.click(await screen.findByRole("button", { name: "日志" }));
    expect(screen.getByText("罗文把玩家提供的说法转告给塔伦；消息仍未验证。")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "旁白" }));
    fireEvent.click(screen.getByRole("button", { name: "自由表达" }));
    fireEvent.change(screen.getByRole("textbox", { name: "向世界旁白表达" }), { target: { value: "这里现在发生了什么？" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(submit).toHaveBeenCalledWith(
      "world_1",
      expect.objectContaining({ target_ids: expect.arrayContaining(["world_narrative", "rowan"]), text: "这里现在发生了什么？", dialogue_interaction_id: expect.any(String) }),
      expect.any(String),
    ));
  });

  it("offers cancellation while a durable command is pending", async () => {
    setupProjectionMocks();
    vi.spyOn(aethelisApi, "submitContextualCommand").mockResolvedValue(receipt("submitted"));
    vi.spyOn(aethelisApi, "getCommand").mockResolvedValue(receipt("submitted"));
    const cancel = vi.spyOn(aethelisApi, "cancelCommand").mockResolvedValue(receipt("cancelled"));
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    fireEvent.click(await screen.findByRole("button", { name: "行动" }));
    fireEvent.click(screen.getByText("前往：中央档案馆"));
    fireEvent.click(screen.getByRole("button", { name: "地图" }));
    expect(screen.getByLabelText("地图面板")).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "取消尚未提交的行动" }));

    await waitFor(() => expect(cancel).toHaveBeenCalledWith("world_1", "command_1"));
    expect(await screen.findByText("行动已取消")).toBeInTheDocument();
  });

  it("refreshes the scene and resubmits after a rejected action", async () => {
    setupProjectionMocks();
    const submit = vi.spyOn(aethelisApi, "submitContextualCommand").mockResolvedValue(receipt("rejected"));
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    fireEvent.click(await screen.findByRole("button", { name: "行动" }));
    fireEvent.click(screen.getByText("前往：中央档案馆"));
    fireEvent.click(await screen.findByRole("button", { name: "刷新现场后重试" }));

    await waitFor(() => expect(submit).toHaveBeenCalledTimes(2));
    expect(aethelisApi.getScene).toHaveBeenCalledTimes(2);
  });

  it("submits free-form intent with visible scene targets", async () => {
    setupProjectionMocks();
    const submit = vi.spyOn(aethelisApi, "submitNaturalLanguageCommand").mockResolvedValue(receipt("needs_clarification"));
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    await openRowanDialogue();
    fireEvent.click(screen.getByRole("button", { name: "自由表达" }));
    const input = screen.getByRole("textbox", { name: "对罗文·凯斯特说" });
    fireEvent.change(input, { target: { value: "问问罗文关于这里的情况" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(submit).toHaveBeenCalledWith(
      "world_1",
      expect.objectContaining({ text: "问问罗文关于这里的情况", target_ids: ["rowan"], target_hints: { rowan: "罗文·凯斯特" }, dialogue_interaction_id: expect.any(String) }),
      expect.any(String),
    ));
  });

  it("turns material ambiguity into a player-facing clarification", async () => {
    setupProjectionMocks();
    const ambiguous = receipt("needs_clarification");
    ambiguous.execution.parsed_intent = { normalized_action: "ask_character", target_ids: [], confidence: 0.4, missing_fields: ["target"], safety_classification: "requires_governance" };
    vi.spyOn(aethelisApi, "submitNaturalLanguageCommand").mockResolvedValue(ambiguous);
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    await openRowanDialogue();
    fireEvent.click(screen.getByRole("button", { name: "自由表达" }));
    fireEvent.change(screen.getByRole("textbox", { name: "对罗文·凯斯特说" }), { target: { value: "去问问" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText(/请补充：对象/)).toBeInTheDocument();
    expect(screen.queryByText("世界回应")).not.toBeInTheDocument();
  });

  it("keeps dialogue controls outside the dialogue box and saves only free-expression history", async () => {
    setupProjectionMocks();
    vi.mocked(aethelisApi.getJournal).mockResolvedValue({
      ...journal,
      dialogue_interactions: [{
        id: "fixed_only",
        target_kind: "character",
        target_id: "rowan",
        target_name: "罗文·凯斯特",
        contains_free_expression: false,
        exchanges: [{ id: "fixed_turn", input_kind: "preset", player_text: "询问守卫情况", response_text: "守卫仍在巡查。", requested_effect_status: "none", visible_effects: [], committed_event_id: "event_fixed" }],
      }, {
        id: "free_history",
        target_kind: "character",
        target_id: "rowan",
        target_name: "罗文·凯斯特",
        contains_free_expression: true,
        exchanges: [{ id: "free_turn", input_kind: "free", player_text: "你相信谁？", response_text: "我更相信可验证的证据。", requested_effect_status: "none", visible_effects: [], committed_event_id: "event_free" }],
      }],
    });
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    await openRowanDialogue();
    const freeButton = screen.getByRole("button", { name: "自由表达" });
    const dialogueBox = screen.getByText("选择一个预设话题，或用自己的话开始交谈。").closest(".conversation-dialogue-box");
    expect(freeButton.closest(".conversation-mode-controls")).toBeInTheDocument();
    expect(dialogueBox?.contains(freeButton)).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "预设选项" }));
    expect(screen.getByText("当前没有可用的预设选项。").closest(".conversation-choice-list")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "历史记录" }));
    expect(screen.getByText("你相信谁？")).toBeInTheDocument();
    expect(screen.queryByText("询问守卫情况")).not.toBeInTheDocument();
  });

  it("explains legacy dialogue capability instead of submitting a guaranteed rejection", async () => {
    setupProjectionMocks();
    vi.mocked(aethelisApi.getScene).mockResolvedValue({
      ...scene,
      content_version_id: "mistgate_product_v1_7_0",
      supports_free_dialogue: false,
      supports_world_narrative: false,
      recommended_content_version_id: "mistgate_product_v1_10_0",
    });
    const submit = vi.spyOn(aethelisApi, "submitNaturalLanguageCommand");
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    await openRowanDialogue();
    expect(screen.getByRole("button", { name: "自由表达" })).toBeDisabled();
    expect(screen.getByText(/这条旧时间线不具备自由对话能力/)).toHaveTextContent("mistgate_product_v1_10_0");
    expect(submit).not.toHaveBeenCalled();
  });

  it("exposes real inventory and world-derived quest overlays", async () => {
    setupProjectionMocks();
    render(<PlayView worldId="world_1" profile={profile} initialSession={session} onExit={vi.fn()} onTimelineChanged={vi.fn().mockResolvedValue(undefined)} />);

    fireEvent.click(await screen.findByRole("button", { name: "背包" }));
    expect(screen.getByText("稳定器零件")).toBeInTheDocument();
    expect(screen.queryByText("维修记录")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "任务" }));
    expect(screen.getByText("选择第一条线索。")).toBeInTheDocument();
    expect(screen.getByText("查阅旧维修记录")).toBeInTheDocument();
  });
});
