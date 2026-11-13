"""Runner owns the terminal receipt; the first persisted answer wins on checkpoint replay."""
from orion.schemas.reading_resonance import ReadingResonanceResultV1


async def persist_resonance(pool, result: ReadingResonanceResultV1) -> None:
    from psycopg.types.json import Jsonb

    async with pool.connection() as conn:
        row = await (await conn.execute(
            "UPDATE reading_resonance SET result_json=%s, completed_at=%s "
            "WHERE seed_id=%s AND run_id=%s AND result_json IS NULL RETURNING run_id",
            (Jsonb(result.model_dump(mode="json")), result.completed_at, result.seed_id, result.run_id),
        )).fetchone()
        if row is None:
            row = await (await conn.execute(
                "SELECT result_json FROM reading_resonance WHERE seed_id=%s AND run_id=%s",
                (result.seed_id, result.run_id))).fetchone()
            if row is None or row["result_json"] is None:
                raise RuntimeError("resonance_request_missing")
