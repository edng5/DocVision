"""Plots for the notebook: architecture diagram, tagged pages, confusion matrix, training curves."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw


def draw_architecture():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, text=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))
        if text:
            ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, text, fontsize=8, color="#555")

    IN, BB, PT, CL, IE = "#e8f1fb", "#fff2cc", "#f4e1f7", "#e2f5e1", "#fde3d6"
    box(0.2, 7.6, 2.6, 0.9, "Page JPG\n(224×224, 16-px patches)", IN)
    box(0.2, 6.4, 2.6, 0.9, "OCR TSV (Textract layout)\nwords + boxes (+ Tag)", IN)
    ax.add_patch(FancyBboxPatch((3.3, 5.6), 6.4, 3.2, boxstyle="round,pad=0.05", fc="#fffaf0", ec="#b8860b", lw=1.5, ls="--"))
    ax.text(6.5, 8.6, "LayoutLMv3-base backbone (125M) — pretrained once, frozen, loaded once", ha="center",
            fontsize=10, weight="bold")
    box(3.5, 7.4, 2.8, 0.8, "Patch embed (linear)\n196 patch tokens + [CLS]", BB)
    box(3.5, 6.2, 2.8, 0.8, "Text embed: BPE tokens\n+ 1D pos + 2D layout (boxes)", BB)
    box(6.8, 6.2, 2.7, 2.0, "12-layer Transformer\n(joint text + image\nself-attention,\n1D/2D relative bias)", BB)
    arrow(2.8, 8.05, 3.5, 7.8); arrow(2.8, 6.85, 3.5, 6.6)
    arrow(6.3, 7.8, 6.8, 7.4); arrow(6.3, 6.6, 6.8, 6.9)
    box(10.2, 7.6, 4.5, 0.9, "MLM head (tied to word emb.)\nspan-masked whole words → tokens", PT)
    box(10.2, 6.4, 4.5, 0.9, "MIM head (linear → 16×16×3 px)\n40% patches → mask token → L1 pixels", PT)
    arrow(9.5, 7.6, 10.2, 8.0, "pretrain"); arrow(9.5, 6.8, 10.2, 6.85)
    box(0.4, 3.6, 3.6, 1.1, "+ LoRA 'cls' (q,k,v)\npage vector =\n[text CLS ; mean text ; image CLS]", CL)
    box(4.4, 3.6, 3.5, 1.1, "+ page-position embedding\nTransformer / BiLSTM\nover ALL pages of the bundle", CL)
    box(8.3, 3.6, 2.9, 1.1, "per-page doc type\n+ boundary (new sub-doc)\nfirst pages weighted", CL)
    arrow(6.5, 5.6, 2.2, 4.7, "frozen backbone"); arrow(4.0, 4.15, 4.4, 4.15); arrow(7.9, 4.15, 8.3, 4.15)
    box(0.4, 1.2, 3.6, 1.2, "+ LoRA 'ie_<doc_type>' (q,k,v)\none adapter per doc type\n(switch = flip active adapter)", IE)
    box(4.4, 1.2, 3.5, 1.2, "Linear head per word\n(first sub-token)\nplain tags: O | field names", IE)
    box(8.3, 1.2, 2.9, 1.2, "sliding windows → merge\nequal-tag runs → entities\n→ document fields", IE)
    ax.plot([3.3, 0.12, 0.12], [5.9, 5.9, 1.8], color="#333", lw=1.3)
    arrow(0.12, 1.8, 0.4, 1.8); arrow(4.0, 1.8, 4.4, 1.8); arrow(7.9, 1.8, 8.3, 1.8)
    arrow(9.75, 3.6, 9.75, 2.4, "segments → which\nextractor to run")
    box(11.8, 2.3, 3.0, 1.6, "Output JSON\npages: type, conf, boundary\nsegments: fields\n(value, conf, page, bbox)", "#eeeeee")
    arrow(11.2, 4.15, 11.8, 3.3); arrow(11.2, 1.8, 11.8, 2.7)
    ax.text(0.2, 0.4, "Stage 1: self-supervised pretraining (MLM + MIM) → Stage 2: LoRA adapters + heads per task "
            "(backbone frozen) → Serving: one backbone in memory, tasks switch by adapter", fontsize=9, style="italic")
    plt.tight_layout()
    plt.show()


def _scaled(box, size):
    return [v * s / 1000 for v, s in zip(box, size * 2)]


def show_tagged_pages(pages, max_pages=4):
    """Draw each page with its labelled field words boxed (one colour per field)."""
    pages = pages[:max_pages]
    fig, axes = plt.subplots(1, len(pages), figsize=(4.6 * len(pages), 6), squeeze=False)
    for ax, p in zip(axes[0], pages):
        img = Image.open(p._image).convert("RGB")
        d, palette = ImageDraw.Draw(img), {}
        for w in p.words:
            if w["tag"] != "O":
                c = palette.setdefault(w["tag"], tuple(int(v * 255) for v in plt.cm.tab10(len(palette) % 10)[:3]))
                d.rectangle(_scaled(w["box"], img.size), outline=c, width=2)
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"{p.meta.get('doc_id')} p{p.meta.get('page_num')} · {p.doc_type}", fontsize=10)
    plt.tight_layout(); plt.show()


def show_predictions(pages, result, max_pages=4):
    """Draw predicted field boxes (red) and predicted vs true page type."""
    seg_of = {pn: s for s in result["segments"] for pn in s["pages"]}
    pages = pages[:max_pages]
    fig, axes = plt.subplots(1, len(pages), figsize=(5 * len(pages), 6.5), squeeze=False)
    for ax, p, pp in zip(axes[0], pages, result["pages"]):
        img = Image.open(p._image).convert("RGB")
        d = ImageDraw.Draw(img)
        for v in (seg_of[pp["page_num"]]["fields"] or {}).values():
            for e in (v if isinstance(v, list) else [v] if v else []):
                if e["page"] == pp["page_num"]:
                    d.rectangle(_scaled(e["bbox"], img.size), outline=(220, 40, 40), width=3)
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"p{pp['page_num']}: pred {pp['doc_type']} ({pp['confidence']:.2f})\ntrue {p.doc_type}", fontsize=9)
    plt.suptitle("Predicted fields (red boxes)"); plt.tight_layout(); plt.show()


def plot_confusion(cm, labels, title="Page classification — confusion matrix"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right"); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.show()


def plot_training(pretrain_hist, clf_hist, ie_hists):
    sm = lambda x, k=5: np.convolve(x, np.ones(k) / k, mode="valid") if len(x) > k else x
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.6))
    axes[0].plot(sm(pretrain_hist["mlm"]), label="MLM loss"); axes[0].plot(sm(pretrain_hist["mim"]), label="MIM L1")
    axes[0].legend(); axes[0].set_title("Stage 1: backbone pretraining"); axes[0].set_xlabel("step")
    axes[1].plot(sm(clf_hist["loss"], 10)); axes[1].set_title("Classifier loss (per bundle)"); axes[1].set_xlabel("step")
    axes[2].plot(range(1, len(clf_hist["val_acc"]) + 1), clf_hist["val_acc"], "o-"); axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Classifier val page accuracy"); axes[2].set_xlabel("epoch")
    for dt, h in ie_hists.items():
        axes[3].plot(range(1, len(h["loss"]) + 1), h["loss"], "o-", label=dt)
    axes[3].legend(fontsize=7); axes[3].set_title("Extractor loss"); axes[3].set_xlabel("epoch")
    plt.tight_layout(); plt.show()
