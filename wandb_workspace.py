"""A saved W&B view (workspace) for the grid-diffusion runs (train_gs.py, diffusion mode), built with wandb-workspaces.

  python wandb_workspace.py                  # print the layout, write nothing
  python wandb_workspace.py save=true        # save it as a NEW view in the project and print its URL

Sections, in order: the full sample's image quality (the comparable generation metric, as eval_diffusion's), the one-step
render at each shared timestep of diffusion.val_t, the validation losses, the train losses, the x0 loss per field / per
timestep bucket (train), media, and bookkeeping (validation sizes, standard errors). The runset is a regex on the run name
(`query`), so runs that match later show up without re-saving. Keys are the ones train_gs.py logs; which ones are comparable
between runs is noted in the section headers (validation loss/x0 and the gated loss/photom follow diffusion.schedule.phase).
Needs `pip install wandb-workspaces` and a W&B login for save=true.
"""
import hydra
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from omegaconf import DictConfig

FIELDS = ("opacity", "scale", "rotation", "sh_dc", "offset")      # gs_diffusion.FIELDS (not imported: keeps this torch-free)
T_BUCKETS = ("t0-0.25", "t0.25-0.5", "t0.5-0.75", "t0.75-1")
SET_TITLES = {"val_meshes": "unseen meshes", "val_views_sval": "unseen views (sval)", "val_views_strain": "unseen views (strain)"}


def line(title, ys, x, smoothing=0.0, **kw):
  ys = [ys] if isinstance(ys, str) else list(ys)
  if smoothing:
    kw.update(smoothing_type="exponential", smoothing_factor=float(smoothing), smoothing_show_original=False)
  return wr.LinePlot(title=title, x=x, y=ys, **kw)


def per_set(cfg, key, title):
  """One panel per validation set for <set>/<validation>/<key>."""
  return [line(f"{title}: {SET_TITLES.get(s, s)}", f"{s}/{cfg.validation}/{key}", cfg.x_axis) for s in cfg.sets]


def section(name, panels, columns=3, is_open=True):
  return ws.Section(name=name, panels=panels, is_open=is_open,
                    layout_settings=ws.SectionLayoutSettings(columns=columns, rows=max(1, -(-len(panels) // columns))))


def build(cfg):
  x, sm = cfg.x_axis, cfg.train_smoothing
  ts = [f"{float(t):g}" for t in cfg.val_t]
  sections = [
    section("Samples: full rollout (comparable across runs)",
            per_set(cfg, "loss/photom_full_fg_psnr", "fg PSNR")
            + per_set(cfg, "loss/photom_full_psnr", "PSNR")
            + per_set(cfg, "loss/photom_full_bg_l1", "bg L1 (haze)")),
    section("One step at a shared t (comparable across runs; fg PSNR, t = 1: 1-step sample)",
            [line(f"t = {t}: {SET_TITLES.get(s, s)}", f"{s}/{cfg.validation}/loss/photom_t{t}_fg_psnr", x)
             for t in ts for s in cfg.sets]),
    section("Validation losses (loss/photom: the training's rendering term; loss/x0 and loss/photom follow schedule.phase)",
            per_set(cfg, "loss/photom_full", "full sample L1+D-SSIM")
            + per_set(cfg, "loss/photom", "rendering term")
            + per_set(cfg, "loss/x0", "x0 loss")),
    section("Training", [
      line("train/loss (weighted total)", "train/loss", x, sm),
      line("train/loss/x0 (unweighted)", "train/loss/x0", x, sm),
      line("train/loss/photom (unweighted)", "train/loss/photom", x, sm),
      line("train fg PSNR of the x0 prediction", "train/loss/photom_fg_psnr", x, sm),
      line("photom_frac (volumes in the gate)", "train/loss/photom_frac", x, sm),
      line("spike-skipped steps", "train/skipped", x, sm),
      line("spike filter average", "train/loss_ema", x),
      line("learning rate (pg1, pg2: the param groups)", ["lr-AdamW/pg1", "lr-AdamW/pg2"], x),
    ], columns=4),
    section("x0 loss per field and per t bucket (train)", [
      line("x0 loss per field", [f"train/loss/x0/{f}" for f in FIELDS], x, sm),
      line("x0 loss per t bucket", [f"train/loss/x0/{b}" for b in T_BUCKETS], x, sm),
    ], columns=2, is_open=False),
    section("Media", [
      wr.MediaBrowser(title="orbits", media_keys=[f"{s}/orbit" for s in ("train", *cfg.sets)]),
      wr.MediaBrowser(title="panels [GT | render | diff]", media_keys=[f"{s}/panel" for s in ("train", *cfg.sets)]),
      wr.MediaBrowser(title="scale layers", media_keys=[f"{s}/scale_layers" for s in ("train", *cfg.sets)]),
    ], columns=3, is_open=False),
    section("Bookkeeping (validation sizes, standard error of the full-sample loss)",
            per_set(cfg, "n_volumes", "volumes scored") + per_set(cfg, "loss_se", "SE of loss"), is_open=False),
  ]
  return ws.Workspace(
    entity=cfg.entity, project=cfg.project, name=cfg.name, sections=sections,
    settings=ws.WorkspaceSettings(x_axis=x, max_runs=int(cfg.max_runs), smoothing_type="none", group_by_prefix="first"),
    runset_settings=ws.RunsetSettings(query=cfg.query, regex_query=True))


@hydra.main(version_base=None, config_path="conf", config_name="wandb_workspace")
def main(cfg: DictConfig):
  if cfg.entity is None:
    import wandb
    cfg.entity = wandb.Api().default_entity
  view = build(cfg)
  for s in view.sections:
    print(f"[{s.name}]")
    for p in s.panels:
      print("  ", getattr(p, "title", None), getattr(p, "y", getattr(p, "media_keys", "")))
  if cfg.save:
    view.save()
    print("saved:", view.url)


if __name__ == "__main__":
  main()
