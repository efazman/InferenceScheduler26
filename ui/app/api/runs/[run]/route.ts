import { NextResponse } from "next/server";
import { readEvents, runPath } from "@/lib/runs";

export const dynamic = "force-dynamic";

export async function GET(_req: Request, ctx: { params: Promise<{ run: string }> }) {
  const { run } = await ctx.params;
  const dir = runPath(run);
  if (!dir) return NextResponse.json({ error: "invalid run name" }, { status: 400 });
  try {
    const events = await readEvents(dir);
    const done = events.find((e) => e.event_type === "run_completed");
    return NextResponse.json({ name: run, events, summary: done?.data?.summary ?? null, complete: Boolean(done) });
  } catch {
    return NextResponse.json({ error: "run not found" }, { status: 404 });
  }
}
