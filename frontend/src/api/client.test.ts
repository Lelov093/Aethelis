import { AethelisApiClient, AethelisApiError } from "./client";

describe("AethelisApiClient", () => {
  it("reads the persisted local profile without an Authorization header", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ id: "profile_local_player", display_name: "雾门旅人" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const client = new AethelisApiClient("http://127.0.0.1:8000/api/v1");
    const profile = await client.getProfile();

    expect(profile.display_name).toBe("雾门旅人");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/me",
      expect.objectContaining({ headers: undefined }),
    );
  });

  it("preserves Product API problem details", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          type: "urn:aethelis:problem:world_archived",
          title: "Aethelis request failed",
          status: 409,
          code: "world_archived",
          detail: "Archived worlds cannot accept commands.",
        }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      ),
    );

    const client = new AethelisApiClient("http://127.0.0.1:8000/api/v1");
    await expect(client.createSave("world_1", "检查点")).rejects.toMatchObject({
      problem: expect.objectContaining({ code: "world_archived", status: 409 }),
    } satisfies Partial<AethelisApiError>);
  });

  it("submits a contextual action with the durable idempotency contract", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ command: { id: "command_1", status: "submitted" } }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const client = new AethelisApiClient("http://127.0.0.1:8000/api/v1");

    await client.submitContextualCommand(
      "world_1",
      {
        player_profile_id: "profile_1",
        play_session_id: "session_1",
        action_id: "move_to_location",
        actor_id: "profile_1",
        target_ids: ["central_archive"],
        location_id: "council_square",
        expected_world_version: 2,
        locale: "zh-CN",
      },
      "move-request-0001",
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/world-instances/world_1/commands",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Idempotency-Key": "move-request-0001" }),
      }),
    );
    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(request.body as string)).toMatchObject({
      input_mode: "contextual_action",
      action_id: "move_to_location",
      expected_world_version: 2,
    });
  });

  it("submits natural language through the same durable command endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ command: { id: "command_nl", status: "submitted" } }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const client = new AethelisApiClient("http://127.0.0.1:8000/api/v1");

    await client.submitNaturalLanguageCommand("world_1", {
      player_profile_id: "profile_1",
      play_session_id: "session_1",
      text: "问问罗文这里发生了什么",
      actor_id: "profile_1",
      target_ids: ["rowan"],
      target_hints: { rowan: "罗文·凯斯特" },
      location_id: "council_square",
      expected_world_version: 2,
      locale: "zh-CN",
    }, "natural-request-0001");

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(request.body as string)).toMatchObject({
      input_mode: "natural_language_intent",
      text: "问问罗文这里发生了什么",
      target_ids: ["rowan"],
    });
  });

  it("loads the connected player projections for one world", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      new Response(JSON.stringify({ world_instance_id: "world_1", world_version: 3 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const client = new AethelisApiClient("http://127.0.0.1:8000/api/v1");

    await Promise.all([
      client.getScene("world_1", "profile 本地"),
      client.getMap("world_1", "profile 本地"),
      client.getJournal("world_1", "profile 本地"),
      client.getResumeSummary("world_1", "profile 本地"),
    ]);

    expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual([
      "http://127.0.0.1:8000/api/v1/world-instances/world_1/scene?player_profile_id=profile%20%E6%9C%AC%E5%9C%B0",
      "http://127.0.0.1:8000/api/v1/world-instances/world_1/map?player_profile_id=profile%20%E6%9C%AC%E5%9C%B0",
      "http://127.0.0.1:8000/api/v1/world-instances/world_1/journal?player_profile_id=profile%20%E6%9C%AC%E5%9C%B0",
      "http://127.0.0.1:8000/api/v1/world-instances/world_1/resume-summary?player_profile_id=profile%20%E6%9C%AC%E5%9C%B0",
    ]);
  });
});
