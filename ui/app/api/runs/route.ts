import { NextResponse } from "next/server";
import { listRuns, RUNS_DIR } from "@/lib/runs";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ runs_dir: RUNS_DIR, runs: await listRuns() });
}
