from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent
SUBMISSION_ROOT = PROJECT_ROOT / "final_submission" / "journal-finder-project"

OUTPUT_FILES = (
    "dataset_summary.csv",
    "recommendation_metrics.csv",
    "top_journals_sample.csv",
    "k_selection_results.csv",
    "final_k_comparison.csv",
    "cluster_summary.csv",
    "cluster_results.csv",
    "alternative_method_summary.csv",
)

FIGURE_FILES = (
    "top_10_journals.png",
    "abstract_length_dist.png",
    "kneedle_elbow.png",
    "silhouette_scores.png",
    "davies_bouldin_scores.png",
    "calinski_harabasz_scores.png",
    "cluster_distribution.png",
    "cluster_visualization.png",
)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path):
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def main():
    if SUBMISSION_ROOT.exists():
        shutil.rmtree(SUBMISSION_ROOT)
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)

    for relative_path in (
        "README.md",
        "main.py",
        "create_notebook.py",
        "build_submission.py",
        "requirements.txt",
        "requirements-bertopic.txt",
        "notebooks/journal_finder_project.ipynb",
        "data/CompSciencePub.sqlite",
        "report/report.tex",
        "report/report.pdf",
        "report/IEEEtran.cls",
    ):
        copy_file(PROJECT_ROOT / relative_path, SUBMISSION_ROOT / relative_path)

    copy_tree(PROJECT_ROOT / "src", SUBMISSION_ROOT / "src")
    for junk_dir in SUBMISSION_ROOT.rglob("__pycache__"):
        shutil.rmtree(junk_dir, ignore_errors=True)
    for junk_file in SUBMISSION_ROOT.rglob(".DS_Store"):
        junk_file.unlink(missing_ok=True)
    for junk_file in SUBMISSION_ROOT.rglob("*.pyc"):
        junk_file.unlink(missing_ok=True)

    bertopic_dst = SUBMISSION_ROOT / "bertopic_experiment"
    bertopic_dst.mkdir(parents=True, exist_ok=True)
    copy_file(PROJECT_ROOT / "bertopic_experiment" / "README.md", bertopic_dst / "README.md")
    copy_file(
        PROJECT_ROOT / "bertopic_experiment" / "run_bertopic_experiment.py",
        bertopic_dst / "run_bertopic_experiment.py",
    )

    outputs_dst = SUBMISSION_ROOT / "outputs"
    outputs_dst.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        copy_file(PROJECT_ROOT / "outputs" / filename, outputs_dst / filename)

    figures_dst = outputs_dst / "figures"
    figures_dst.mkdir(parents=True, exist_ok=True)
    for filename in FIGURE_FILES:
        copy_file(PROJECT_ROOT / "outputs" / "figures" / filename, figures_dst / filename)

    print(f"Clean submission created at: {SUBMISSION_ROOT}")


if __name__ == "__main__":
    main()
