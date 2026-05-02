# Demo Checklist

Before recording:

- Replace `<your-github-username>` after the repo is public.
- Replace `TODO_PUBLIC_VIDEO_URL` after uploading the presentation.
- Recompile `paper/final_project.pdf` after URL changes.
- Confirm `python -m src.run_all` completes.
- Confirm `results/metrics.csv` exists.
- Confirm all major figures exist in `figures/`.
- Open `figures/figure_contact_sheet.png` and visually inspect it.
- Open `paper/final_project.pdf`.
- Keep raw data caches out of git unless intentionally needed.

During recording:

- Start with the research question.
- State that Yahoo Finance data are public and the original work is the reproducible pipeline.
- Explain the future-volatility target and leakage prevention.
- Show the chronological train/validation/test split.
- Show the model comparison table.
- Show the main figures.
- State the honest LSTM limitation and identify the reported neural row as an MLP fallback.
- End with the one-command reproduction command.

After recording:

- Upload the video.
- Make the video link public or instructor-accessible.
- Replace `TODO_PUBLIC_VIDEO_URL` in `README.md` and `paper/main.tex`.
- Recompile the paper.
