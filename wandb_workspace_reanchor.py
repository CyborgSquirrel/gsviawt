"""A saved W&B view (workspace) for the re-anchoring runs (train_reanchor.py: A's grid + B's point cloud -> B's grid), built with
wandb-workspaces, like wandb_workspace.py for the diffusion runs.

  python wandb_workspace_reanchor.py                         # print the layout, write nothing
  python wandb_workspace_reanchor.py save=true               # save it as a NEW view in the project and print its URL
  python wandb_workspace_reanchor.py save=true url=<view URL> # update that view in place (panels + settings; its runset is kept)

Sections: for the full-dataset runs (data.dataset full) one per eval set (val, shard2, train_heldout: eval/<set>/* with its
references ref/<set>/*, and on val the A-ablated eval); then the one-mesh runs' panels: held-out PSNR (views 45-49, over white and over black) with the three references as lines in the same panel (A as-is,
B's own fit = ceiling, matched copy of A's Gaussians into B's slots = floor); the same with A ablated at eval time (does the model
read A?); the train losses; bookkeeping (lr, grad norm, step time, memory, token counts). The runset is a regex on the run name.
"""
import hydra
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from omegaconf import DictConfig

REFS = (("A_asis", "A as-is"), ("B_fit", "B's own fit (ceiling)"), ("matched32", "matched copy (floor)"))
FULL_SETS = (("val", "new100 val meshes (held-out views)"), ("shard2", "shard2 unseen meshes (held-out views)"),
             ("train_heldout", "training meshes, held-out views"))


def line(title, ys, x, smoothing=0.0, **kw):
  ys = [ys] if isinstance(ys, str) else list(ys)
  if smoothing:
    kw.update(smoothing_type="exponential", smoothing_factor=float(smoothing), smoothing_show_original=False)
  return wr.LinePlot(title=title, x=x, y=ys, **kw)


def section(name, panels, columns=2, is_open=True):
  return ws.Section(name=name, panels=panels, is_open=is_open,
                    layout_settings=ws.SectionLayoutSettings(columns=columns, rows=max(1, -(-len(panels) // columns))))


def workspace(cfg):
  x, sm = cfg.x_axis, cfg.train_smoothing
  held = [line(f"held-out PSNR over {bg} (model + references)", [f"eval/{bg}", *[f"ref/{k}_{bg}" for k, _ in REFS]], x)
          for bg in ("white", "black")]
  ablate = [line(f"held-out PSNR over {bg}: model vs A ablated at eval", [f"eval/{bg}", f"eval/ablate_a_{bg}"], x)
            for bg in ("white", "black")]
  train = [line("train loss (L1 + D-SSIM, random-colour background)", "train/loss", x, sm),
           line("train PSNR (random-colour background)", "train/psnr", x, sm),
           line("train L1", "train/l1", x, sm), line("train D-SSIM", "train/dssim", x, sm)]
  book = [line("learning rate", "train/lr", x), line("grad norm", "train/grad_norm", x, sm),
          line("seconds per step", "train/step_seconds", x, sm), line("max GPU memory (GB)", "train/max_mem_gb", x),
          line("tokens: B", "train/tokens_b", x, sm), line("tokens: A", "train/tokens_a", x, sm)]
  full = []
  for name, title in FULL_SETS:   # the full-dataset runs (data.dataset full): one section per eval set
    panels = [line(f"{title}: PSNR over {bg} (model + references)", [f"eval/{name}/{bg}", *[f"ref/{name}/{k}_{bg}" for k, _ in REFS]], x)
              for bg in ("white", "black")]
    if name == "val":
      panels += [line(f"{title}: model vs A ablated at eval, over {bg}", [f"eval/{name}/{bg}", f"eval/{name}/ablate_a_{bg}"], x)
                 for bg in ("white", "black")]
    full.append(section(f"full dataset -- {title}", panels))
  book.append(line("data wait per step (s)", "train/data_wait_seconds", x, sm))
  book.append(line("eval seconds", "eval/seconds", x))
  book.append(line("LR cycle", "train/cycle", x))
  sections = [*full,
              section("one mesh: held-out PSNR (views 45-49), with references", held),
              section("one mesh: does it use A? (A ablated at eval)", ablate),
              section("training", train),
              section("bookkeeping", book, columns=3, is_open=False)]
  return ws.Workspace(
    entity=cfg.entity, project=cfg.project, name=cfg.name, sections=sections,
    settings=ws.WorkspaceSettings(x_axis=x, max_runs=int(cfg.max_runs), smoothing_type="none", group_by_prefix="first"),
    runset_settings=ws.RunsetSettings(query=cfg.query, regex_query=True))


@hydra.main(version_base=None, config_path="conf", config_name="wandb_workspace_reanchor")
def main(cfg: DictConfig):
  if cfg.entity is None:
    import wandb
    cfg.entity = wandb.Api().default_entity
  w = workspace(cfg)
  for s in w.sections:
    print(f"[{s.name}]", *[p.title for p in s.panels], sep="\n  ")
  if cfg.save:
    if cfg.url:
      existing = ws.Workspace.from_url(cfg.url)
      existing.sections, existing.settings = w.sections, w.settings
      existing.save()
      print("updated", existing.url)
    else:
      w.save()
      print("saved", w.url)


if __name__ == "__main__":
  main()
