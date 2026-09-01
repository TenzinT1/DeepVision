import React, { useCallback, useEffect, useState } from "react";
import { ApiError, getSettings, putSettings } from "../api/client";
import type {
  AppSettings,
  ModelChoice,
  ModelVendor,
  OCRLanguage,
  ProviderMode,
  ReportDetail,
  SettingsResponse,
} from "../api/types";

/* ==========================================================================
   Settings page — ported from the '===== SETTINGS =====' block of
   .design_extracted/DeepVision.dc.html. Layout, spacing, colors and copy
   match the prototype; data is wired to GET/PUT /api/settings.
   ========================================================================== */

type ProviderKey = "llm" | "vision" | "embedding";

const OCR_OPTIONS: { value: OCRLanguage; label: string }[] = [
  { value: "eng", label: "English" },
  { value: "eng+fra", label: "English + French" },
  { value: "chi_sim", label: "Chinese (Simplified)" },
  { value: "jpn", label: "Japanese" },
  { value: "auto", label: "Auto-detect" },
];

const PROVIDER_LABELS: Record<ProviderKey, { title: string; sub: string }> = {
  llm: { title: "Language model", sub: "LLM" },
  vision: { title: "Vision", sub: "FIGURE ANALYSIS" },
  embedding: { title: "Embeddings", sub: "RETRIEVAL" },
};

/** Human label for the API-key input, keyed by the vendor of the currently
 *  selected model for that stage — so it's obvious which key to paste. */
const VENDOR_KEY_LABELS: Record<ModelVendor, string> = {
  local: "API key",
  anthropic: "Anthropic API key",
  openai: "OpenAI API key",
};

function getMode(s: AppSettings, key: ProviderKey): ProviderMode {
  if (key === "llm") return s.llm_mode;
  if (key === "vision") return s.vision_mode;
  return s.embedding_mode;
}

function getModelId(s: AppSettings, key: ProviderKey): string | null {
  if (key === "llm") return s.llm_model ?? null;
  if (key === "vision") return s.vision_model ?? null;
  return s.embedding_model ?? null;
}

/** Selecting a ModelChoice must move `<stage>_mode` and `<stage>_model`
 *  together — that is the whole point of the model catalog. */
function withChoice(s: AppSettings, key: ProviderKey, choice: ModelChoice): AppSettings {
  switch (key) {
    case "llm":
      return { ...s, llm_mode: choice.mode, llm_model: choice.id };
    case "vision":
      return { ...s, vision_mode: choice.mode, vision_model: choice.id };
    case "embedding":
      return { ...s, embedding_mode: choice.mode, embedding_model: choice.id };
  }
}

/** Local entries first, API entries after — independent of catalog order. */
function groupChoices(choices: ModelChoice[]): { local: ModelChoice[]; api: ModelChoice[] } {
  return {
    local: choices.filter((c) => c.mode === "local"),
    api: choices.filter((c) => c.mode === "api"),
  };
}

function cardStyle(): React.CSSProperties {
  return {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 13,
    padding: 22,
  };
}

function selectStyle(): React.CSSProperties {
  return {
    flex: 1,
    padding: "10px 12px",
    fontSize: 13.5,
    border: "1px solid var(--border-strong)",
    borderRadius: 9,
    background: "var(--surface)",
    color: "var(--fg)",
  };
}

/** The detail-level <select> when the choice is flagged as slow (see
 *  `slowOnLocal` below) — same geometry as `selectStyle()`, danger border. */
function dangerSelectStyle(): React.CSSProperties {
  return { ...selectStyle(), borderColor: "var(--red)" };
}

/** Detail levels that make the local model do materially more generation.
 *  Concise is the recommended level on a local model, so it is not flagged. */
const SLOW_LOCAL_DETAILS: ReportDetail[] = ["standard", "detailed"];

function inputStyle(): React.CSSProperties {
  return {
    width: "100%",
    marginTop: 6,
    padding: "10px 12px",
    fontSize: 13,
    fontFamily: "'IBM Plex Mono'",
    border: "1px solid var(--border-strong)",
    borderRadius: 9,
    background: "var(--surface)",
    color: "var(--fg)",
  };
}

/** Default settings used only if the very first GET fails and the user
 *  wants to see the form shape while retrying (never sent unless saved). */
const FALLBACK_SETTINGS: AppSettings = {
  llm_mode: "local",
  vision_mode: "local",
  embedding_mode: "local",
  llm_model: null,
  vision_model: null,
  embedding_model: null,
  keys: { llm_api_key: null, vision_api_key: null, embedding_api_key: null },
  ocr_language: "eng",
  chunk_size: 800,
  chunk_overlap: 120,
  image_embedding: true,
  report_detail: "standard",
};

export default function SettingsPage() {
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [resp, setResp] = useState<SettingsResponse | null>(null);

  // Editable draft — diverges from `resp.settings` until saved.
  const [draft, setDraft] = useState<AppSettings>(FALLBACK_SETTINGS);
  // New key values typed by the user this session. Empty string = "leave
  // whatever is already saved on the server untouched" (see assumption in
  // the final report: PUT with a null key preserves the stored value).
  const [keyInputs, setKeyInputs] = useState<Record<ProviderKey, string>>({
    llm: "",
    vision: "",
    embedding: "",
  });

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<number | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setLoadError(null);
    getSettings()
      .then((r) => {
        setResp(r);
        setDraft(r.settings);
        setKeyInputs({ llm: "", vision: "", embedding: "" });
      })
      .catch((err) => {
        setLoadError(err instanceof ApiError ? err.message : "Failed to load settings.");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const setKeyInput = (key: ProviderKey, value: string) => {
    setKeyInputs((k) => ({ ...k, [key]: value }));
  };

  const setChunkSize = (n: number) => {
    setDraft((d) => ({
      ...d,
      chunk_size: n,
      chunk_overlap: Math.min(d.chunk_overlap, Math.max(0, n - 20)),
    }));
  };

  const setChunkOverlap = (n: number) => {
    setDraft((d) => ({ ...d, chunk_overlap: Math.min(n, Math.max(0, d.chunk_size - 20)) }));
  };

  const toggleImageEmbedding = () => {
    setDraft((d) => ({ ...d, image_embedding: !d.image_embedding }));
  };

  const needsKeys =
    draft.llm_mode === "api" || draft.vision_mode === "api" || draft.embedding_mode === "api";

  const overlapInvalid = draft.chunk_overlap >= draft.chunk_size;

  const handleReset = () => {
    if (!resp) return;
    setDraft(resp.settings);
    setKeyInputs({ llm: "", vision: "", embedding: "" });
    setSaveError(null);
  };

  const handleSave = async () => {
    if (overlapInvalid || saving) return;
    setSaving(true);
    setSaveError(null);
    const payload: AppSettings = {
      ...draft,
      keys: {
        llm_api_key: keyInputs.llm.trim() ? keyInputs.llm.trim() : null,
        vision_api_key: keyInputs.vision.trim() ? keyInputs.vision.trim() : null,
        embedding_api_key: keyInputs.embedding.trim() ? keyInputs.embedding.trim() : null,
      },
    };
    try {
      const r = await putSettings(payload);
      setResp(r);
      setDraft(r.settings);
      setKeyInputs({ llm: "", vision: "", embedding: "" });
      setSavedAt(Date.now());
      setTimeout(() => setSavedAt(null), 3000);
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Failed to save settings.");
    } finally {
      setSaving(false);
    }
  };

  // ---- vision status pill -------------------------------------------------
  const visionModeDirty = !!resp && draft.vision_mode !== resp.settings.vision_mode;
  let visionBg = "var(--surface-2)";
  let visionDot = "var(--fg-subtle)";
  let visionText = "Checking vision availability…";
  if (loadError) {
    visionBg = "var(--red-soft)";
    visionDot = "var(--red)";
    visionText = "Vision status unavailable — backend unreachable.";
  } else if (visionModeDirty) {
    visionBg = "var(--surface-2)";
    visionDot = "var(--fg-subtle)";
    visionText = "Save settings to check vision availability for this mode.";
  } else if (resp) {
    if (resp.vision_available) {
      visionBg = "var(--green-soft)";
      visionDot = "var(--green)";
    } else if (draft.vision_mode === "api") {
      visionBg = "var(--accent-soft)";
      visionDot = "var(--accent)";
    } else {
      visionBg = "var(--amber-soft)";
      visionDot = "var(--amber)";
    }
    visionText = resp.vision_status_text;
  }

  const toggleTrackStyle: React.CSSProperties = {
    flex: "none",
    width: 42,
    height: 24,
    borderRadius: 20,
    padding: 2,
    transition: "background .15s",
    background: draft.image_embedding ? "var(--accent)" : "var(--border-strong)",
  };
  const toggleKnobStyle: React.CSSProperties = {
    width: 20,
    height: 20,
    borderRadius: "50%",
    background: "#fff",
    transition: "transform .15s",
    transform: `translateX(${draft.image_embedding ? 18 : 0}px)`,
  };

  // ---- loading / error states ---------------------------------------------
  if (loading) {
    return (
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "30px 28px 60px" }}>
        <div style={cardStyle()}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontSize: 13,
              color: "var(--fg-muted)",
            }}
          >
            <span
              style={{
                width: 14,
                height: 14,
                borderRadius: "50%",
                border: "2px solid var(--border-strong)",
                borderTopColor: "var(--accent)",
                animation: "dv-spin .7s linear infinite",
                flex: "none",
              }}
            />
            Loading settings…
          </div>
        </div>
      </div>
    );
  }

  if (loadError && !resp) {
    return (
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "30px 28px 60px" }}>
        <div style={cardStyle()}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--red)" }}>
            Couldn&rsquo;t load settings
          </div>
          <div style={{ fontSize: 12.5, color: "var(--fg-subtle)", marginTop: 6 }}>{loadError}</div>
          <button
            onClick={load}
            style={{
              marginTop: 16,
              fontSize: 13.5,
              fontWeight: 600,
              color: "var(--accent-fg)",
              background: "var(--accent)",
              border: "none",
              borderRadius: 9,
              padding: "9px 18px",
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const keysPresent = resp?.keys_present ?? { llm: false, vision: false, embedding: false };
  const modelCatalog = resp?.model_catalog ?? { llm: [], vision: [], embedding: [] };
  const detailLevels = resp?.detail_levels ?? [];
  const selectedDetailLevel =
    detailLevels.find((lvl) => lvl.value === draft.report_detail) ?? null;
  // Report generation is only slow when the *LLM* stage runs locally — the
  // report agents are the only thing the detail level lengthens. On an API
  // model nothing degrades, so no warning is shown at all.
  const slowOnLocal =
    draft.llm_mode === "local" && SLOW_LOCAL_DETAILS.includes(draft.report_detail);

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: "30px 28px 60px" }}>
      {/* ---- Providers ---- */}
      <div style={cardStyle()}>
        <div style={{ fontSize: 15, fontWeight: 600 }}>Providers</div>
        <div style={{ fontSize: 12.5, color: "var(--fg-subtle)", marginTop: 3 }}>
          Choose a local model (free, runs on your machine) or an external API model for each stage.
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 20 }}>
          {(["llm", "vision", "embedding"] as ProviderKey[]).map((key) => {
            const labels = PROVIDER_LABELS[key];
            const choices = modelCatalog[key];
            const { local, api } = groupChoices(choices);
            const mode = getMode(draft, key);
            const modelId = getModelId(draft, key);
            // Fall back to the first catalog entry matching the current mode
            // so the <select> always has a defined value, even before the
            // user has explicitly picked a model for this stage.
            const selectedId =
              modelId ?? choices.find((c) => c.mode === mode)?.id ?? choices[0]?.id ?? "";
            const selected = choices.find((c) => c.id === selectedId) ?? null;
            return (
              <div key={key}>
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                  <div style={{ width: 150, flex: "none" }}>
                    <div style={{ fontSize: 13.5, fontWeight: 600 }}>{labels.title}</div>
                    <div className="dv-mono" style={{ fontSize: 11, color: "var(--fg-subtle)" }}>
                      {labels.sub}
                    </div>
                  </div>
                  <select
                    value={selectedId}
                    onChange={(e) => {
                      const choice = choices.find((c) => c.id === e.target.value);
                      if (choice) setDraft((d) => withChoice(d, key, choice));
                    }}
                    style={selectStyle()}
                  >
                    {local.length > 0 && (
                      <optgroup label="Local (free)">
                        {local.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.label}
                          </option>
                        ))}
                      </optgroup>
                    )}
                    {api.length > 0 && (
                      <optgroup label="API">
                        {api.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.label}
                          </option>
                        ))}
                      </optgroup>
                    )}
                  </select>
                </div>
                {selected && (
                  <div
                    style={{
                      fontSize: 11.5,
                      color: "var(--fg-subtle)",
                      marginTop: 6,
                      marginLeft: 166,
                    }}
                  >
                    {selected.note}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <div
          style={{
            marginTop: 16,
            padding: "12px 14px",
            borderRadius: 10,
            background: visionBg,
            display: "flex",
            alignItems: "center",
            gap: 9,
          }}
        >
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background: visionDot,
              flex: "none",
            }}
          />
          <span style={{ fontSize: 12.5, color: "var(--fg)" }}>{visionText}</span>
        </div>
      </div>

      {/* ---- API keys ---- */}
      {needsKeys && (
        <div style={{ ...cardStyle(), marginTop: 16 }}>
          <div style={{ fontSize: 15, fontWeight: 600 }}>API keys</div>
          <div style={{ fontSize: 12.5, color: "var(--fg-subtle)", marginTop: 3 }}>
            Only required for stages set to API. Stored locally, never sent anywhere else.
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 18 }}>
            {(["llm", "vision", "embedding"] as ProviderKey[])
              .filter((key) => getMode(draft, key) === "api")
              .map((key) => {
                const present = keysPresent[key];
                const modelId = getModelId(draft, key);
                const choices = modelCatalog[key];
                const selected =
                  choices.find((c) => c.id === modelId) ?? choices.find((c) => c.mode === "api") ?? null;
                const vendor: ModelVendor = selected?.vendor ?? "anthropic";
                return (
                  <label key={key} style={{ display: "block" }}>
                    <span style={{ fontSize: 12.5, fontWeight: 500, color: "var(--fg-muted)" }}>
                      {VENDOR_KEY_LABELS[vendor]}
                    </span>
                    <input
                      type="password"
                      autoComplete="off"
                      value={keyInputs[key]}
                      onChange={(e) => setKeyInput(key, e.target.value)}
                      placeholder={present ? "Key saved — leave blank to keep it" : "sk-••••••••••••••••"}
                      style={inputStyle()}
                    />
                    {vendor === "anthropic" && (
                      <div style={{ fontSize: 11, color: "var(--fg-subtle)", marginTop: 5 }}>
                        From console.anthropic.com, billed separately — a Claude Max/Pro subscription
                        does not grant API access.
                      </div>
                    )}
                  </label>
                );
              })}
          </div>
        </div>
      )}

      {/* ---- Report ---- */}
      <div style={{ ...cardStyle(), marginTop: 16 }}>
        <div style={{ fontSize: 15, fontWeight: 600 }}>Report</div>
        <div style={{ fontSize: 12.5, color: "var(--fg-subtle)", marginTop: 3 }}>
          How long and detailed each generated section should be.
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginTop: 18 }}>
          <div style={{ width: 150, flex: "none" }}>
            <div style={{ fontSize: 13.5, fontWeight: 600 }}>Detail level</div>
          </div>
          <select
            value={draft.report_detail}
            onChange={(e) => setDraft((d) => ({ ...d, report_detail: e.target.value as ReportDetail }))}
            style={slowOnLocal ? dangerSelectStyle() : selectStyle()}
            aria-describedby={slowOnLocal ? "dv-detail-slow-warning" : undefined}
          >
            {detailLevels.map((lvl) => (
              <option key={lvl.value} value={lvl.value}>
                {lvl.label}
              </option>
            ))}
          </select>
        </div>
        {/* Backend-authored length contract (DETAIL_LEVEL_INFO -> detail_levels).
            The speed warning below is *additional* to this, never a replacement. */}
        {selectedDetailLevel && (
          <div style={{ fontSize: 11.5, color: "var(--fg-subtle)", marginTop: 8, marginLeft: 166 }}>
            {selectedDetailLevel.blurb}
          </div>
        )}
        {slowOnLocal && (
          <div
            id="dv-detail-slow-warning"
            role="note"
            style={{
              marginTop: 10,
              marginLeft: 166,
              padding: "10px 12px",
              borderRadius: 9,
              border: "1px solid var(--red)",
              background: "var(--red-soft)",
              fontSize: 11.5,
              lineHeight: 1.55,
              color: "var(--fg)",
            }}
          >
            <div style={{ fontWeight: 600, color: "var(--red)" }}>
              Slow on a local model
            </div>
            <div style={{ marginTop: 3 }}>
              {draft.report_detail === "detailed" ? (
                <>
                  Standard took <strong>~21&nbsp;min</strong> per report measured on
                  llama3.1:8b; Detailed asks the model for roughly twice as many tokens
                  per section, so expect <em>considerably</em> longer — that part is an
                  estimate, not a measurement.
                </>
              ) : (
                <>
                  Standard took <strong>~21&nbsp;min</strong> per report measured on
                  llama3.1:8b. Detailed is considerably longer still (estimate).
                </>
              )}{" "}
              <strong>Concise is recommended</strong> on a weak local model. Switching
              the Language model above to <strong>Claude Sonnet 5</strong> gives full
              quality at any detail level in well under a minute.
            </div>
          </div>
        )}
      </div>

      {/* ---- Extraction ---- */}
      <div style={{ ...cardStyle(), marginTop: 16 }}>
        <div style={{ fontSize: 15, fontWeight: 600 }}>Extraction</div>
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginTop: 18 }}>
          <div style={{ width: 150, flex: "none" }}>
            <div style={{ fontSize: 13.5, fontWeight: 600 }}>OCR language</div>
          </div>
          <select
            value={draft.ocr_language}
            onChange={(e) => setDraft((d) => ({ ...d, ocr_language: e.target.value as OCRLanguage }))}
            style={selectStyle()}
          >
            {OCR_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>

        <div style={{ marginTop: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13.5 }}>
            <span style={{ fontWeight: 600 }}>Chunk size</span>
            <span className="dv-mono" style={{ color: "var(--accent)" }}>
              {draft.chunk_size} tokens
            </span>
          </div>
          <input
            type="range"
            min={256}
            max={2048}
            step={64}
            value={draft.chunk_size}
            onChange={(e) => setChunkSize(Number(e.target.value))}
            style={{ width: "100%", marginTop: 10, accentColor: "var(--accent)" }}
          />
        </div>

        <div style={{ marginTop: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13.5 }}>
            <span style={{ fontWeight: 600 }}>Chunk overlap</span>
            <span className="dv-mono" style={{ color: "var(--accent)" }}>
              {draft.chunk_overlap} tokens
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={400}
            step={20}
            value={draft.chunk_overlap}
            onChange={(e) => setChunkOverlap(Number(e.target.value))}
            style={{ width: "100%", marginTop: 10, accentColor: "var(--accent)" }}
          />
          {overlapInvalid && (
            <div style={{ fontSize: 11.5, color: "var(--red)", marginTop: 6 }}>
              Chunk overlap must be less than chunk size.
            </div>
          )}
        </div>

        <div
          role="switch"
          aria-checked={draft.image_embedding}
          tabIndex={0}
          onClick={toggleImageEmbedding}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              toggleImageEmbedding();
            }
          }}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginTop: 22,
            padding: "13px 15px",
            border: "1px solid var(--border)",
            borderRadius: 10,
            cursor: "pointer",
          }}
        >
          <div>
            <div style={{ fontSize: 13.5, fontWeight: 600 }}>Advanced image embedding</div>
            <div style={{ fontSize: 12, color: "var(--fg-subtle)", marginTop: 2 }}>
              Embed figure crops for visual retrieval. Slower ingestion.
            </div>
          </div>
          <div style={toggleTrackStyle}>
            <div style={toggleKnobStyle} />
          </div>
        </div>
      </div>

      {saveError && (
        <div
          style={{
            marginTop: 16,
            padding: "12px 14px",
            borderRadius: 10,
            background: "var(--red-soft)",
            color: "var(--red)",
            fontSize: 12.5,
          }}
        >
          {saveError}
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 18, justifyContent: "flex-end" }}>
        {savedAt && (
          <span style={{ fontSize: 12.5, color: "var(--green)", marginRight: "auto" }}>Saved</span>
        )}
        <button
          onClick={handleReset}
          disabled={saving}
          style={{
            fontSize: 13.5,
            fontWeight: 500,
            color: "var(--fg-muted)",
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: 9,
            padding: "10px 18px",
            opacity: saving ? 0.6 : 1,
          }}
        >
          Reset defaults
        </button>
        <button
          onClick={handleSave}
          disabled={saving || overlapInvalid}
          style={{
            fontSize: 13.5,
            fontWeight: 600,
            color: "var(--accent-fg)",
            background: "var(--accent)",
            border: "none",
            borderRadius: 9,
            padding: "10px 20px",
            opacity: saving || overlapInvalid ? 0.6 : 1,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          {saving && (
            <span
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                border: "2px solid rgba(255,255,255,.4)",
                borderTopColor: "var(--accent-fg)",
                animation: "dv-spin .7s linear infinite",
              }}
            />
          )}
          {saving ? "Saving…" : "Save settings"}
        </button>
      </div>
    </div>
  );
}
