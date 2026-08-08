import { fireEvent, render, screen } from "@testing-library/react";

import { AppShell } from "./AppShell";

describe("AppShell", () => {
  it("exposes the approved player information architecture", () => {
    const navigate = vi.fn();
    render(
      <AppShell activeView="home" displayName="雾门旅人" apiConnected onNavigate={navigate}>
        <p>世界概览</p>
      </AppShell>,
    );

    expect(screen.getByText("世界概览")).toBeInTheDocument();
    expect(screen.getAllByText("开始新旅程").length).toBeGreaterThan(0);
    expect(screen.getAllByText("存档与时间线").length).toBeGreaterThan(0);
    expect(screen.getByText("本地世界已连接")).toBeInTheDocument();

    fireEvent.click(screen.getAllByText("存档与时间线")[0]);
    expect(navigate).toHaveBeenCalledWith("timelines");
  });

  it("gives active play the full viewport without product navigation chrome", () => {
    render(
      <AppShell activeView="play" displayName="雾门旅人" apiConnected onNavigate={vi.fn()}>
        <p>可移动世界</p>
      </AppShell>,
    );

    expect(screen.getByText("可移动世界")).toBeInTheDocument();
    expect(screen.queryByLabelText("产品导航")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("移动端产品导航")).not.toBeInTheDocument();
  });
});
