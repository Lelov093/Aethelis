import type {
  AvailableWorldContent,
  CommandReceipt,
  ContextualCommandInput,
  NaturalLanguageCommandInput,
  JournalView,
  MapView,
  PlayerProfile,
  PlaySession,
  ProblemDetail,
  ResumeState,
  ResumeSummaryView,
  SavePoint,
  SavePointView,
  SceneView,
  WorldInstance,
  WorldTimelineView,
} from "./contracts";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/api/v1";

export class AethelisApiError extends Error {
  readonly problem: ProblemDetail;

  constructor(problem: ProblemDetail) {
    super(problem.detail);
    this.name = "AethelisApiError";
    this.problem = problem;
  }
}

export class AethelisApiClient {
  constructor(
    private readonly baseUrl =
      import.meta.env.VITE_AETHELIS_API_BASE_URL?.replace(/\/$/, "") ?? DEFAULT_API_BASE_URL,
  ) {}

  getProfile(): Promise<PlayerProfile> {
    return this.request("/me");
  }

  listWorldContent(): Promise<AvailableWorldContent[]> {
    return this.request("/world-definitions");
  }

  listTimelines(includeArchived = false): Promise<WorldTimelineView[]> {
    return this.request(`/world-instances?include_archived=${includeArchived}`);
  }

  createTimeline(input: {
    content_version_id: string;
    player_profile_id: string;
    name: string;
  }): Promise<ResumeState> {
    return this.request("/world-instances", { method: "POST", body: JSON.stringify(input) });
  }

  startSession(worldId: string, playerProfileId: string): Promise<PlaySession> {
    return this.request(`/world-instances/${worldId}/sessions`, {
      method: "POST",
      body: JSON.stringify({ player_profile_id: playerProfileId }),
    });
  }

  getScene(worldId: string, playerProfileId: string): Promise<SceneView> {
    return this.request(`/world-instances/${worldId}/scene?player_profile_id=${encodeURIComponent(playerProfileId)}`);
  }

  getMap(worldId: string, playerProfileId: string): Promise<MapView> {
    return this.request(`/world-instances/${worldId}/map?player_profile_id=${encodeURIComponent(playerProfileId)}`);
  }

  getJournal(worldId: string, playerProfileId: string): Promise<JournalView> {
    return this.request(`/world-instances/${worldId}/journal?player_profile_id=${encodeURIComponent(playerProfileId)}`);
  }

  getResumeSummary(worldId: string, playerProfileId: string): Promise<ResumeSummaryView> {
    return this.request(`/world-instances/${worldId}/resume-summary?player_profile_id=${encodeURIComponent(playerProfileId)}`);
  }

  submitContextualCommand(
    worldId: string,
    input: ContextualCommandInput,
    idempotencyKey: string,
  ): Promise<CommandReceipt> {
    return this.request(`/world-instances/${worldId}/commands`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body: JSON.stringify({ ...input, input_mode: "contextual_action" }),
    });
  }

  submitNaturalLanguageCommand(
    worldId: string,
    input: NaturalLanguageCommandInput,
    idempotencyKey: string,
  ): Promise<CommandReceipt> {
    return this.request(`/world-instances/${worldId}/commands`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body: JSON.stringify({ ...input, input_mode: "natural_language_intent" }),
    });
  }

  getCommand(worldId: string, commandId: string): Promise<CommandReceipt> {
    return this.request(`/world-instances/${worldId}/commands/${commandId}`);
  }

  cancelCommand(worldId: string, commandId: string): Promise<CommandReceipt> {
    return this.request(`/world-instances/${worldId}/commands/${commandId}/cancel`, {
      method: "POST",
    });
  }

  listSaves(worldId: string): Promise<SavePointView[]> {
    return this.request(`/world-instances/${worldId}/saves`);
  }

  createSave(worldId: string, name: string): Promise<SavePoint> {
    return this.request(`/world-instances/${worldId}/saves`, {
      method: "POST",
      body: JSON.stringify({ name }),
    });
  }

  forkSave(
    worldId: string,
    saveId: string,
    playerProfileId: string,
    name: string,
  ): Promise<ResumeState> {
    return this.request(`/world-instances/${worldId}/saves/${saveId}/fork`, {
      method: "POST",
      body: JSON.stringify({ player_profile_id: playerProfileId, name }),
    });
  }

  archiveTimeline(worldId: string): Promise<WorldInstance> {
    return this.request(`/world-instances/${worldId}/archive`, { method: "POST" });
  }

  restoreTimeline(worldId: string): Promise<WorldInstance> {
    return this.request(`/world-instances/${worldId}/restore`, { method: "POST" });
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: init?.body ? { "Content-Type": "application/json", ...init.headers } : init?.headers,
      });
    } catch {
      throw new AethelisApiError({
        type: "urn:aethelis:problem:api_unreachable",
        title: "无法连接 Aethelis",
        status: 0,
        code: "api_unreachable",
        detail: "本地 Product API 尚未启动，请运行 scripts/dev.ps1 后重试。",
      });
    }
    if (!response.ok) {
      const fallback: ProblemDetail = {
        type: "urn:aethelis:problem:unexpected_response",
        title: "请求失败",
        status: response.status,
        code: "unexpected_response",
        detail: `Product API 返回 ${response.status}。`,
      };
      throw new AethelisApiError(
        (await response.json().catch(() => fallback)) as ProblemDetail,
      );
    }
    return (await response.json()) as T;
  }
}

export const aethelisApi = new AethelisApiClient();
