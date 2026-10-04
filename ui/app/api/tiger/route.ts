import { NextResponse } from "next/server";
import { tigerHistory } from "@/lib/tiger";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(await tigerHistory());
}
