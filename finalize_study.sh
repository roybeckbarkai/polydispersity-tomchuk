#!/bin/bash
set -e

# Find the most recent FULL study directory
STUDY_DIR=$(ls -dt studies/tenor_vs_tomchuk_full_* | head -1)

if [ -z "$STUDY_DIR" ] || [ ! -d "$STUDY_DIR" ]; then
    echo "Could not find a full study directory."
    exit 1
fi

echo "Found study directory: $STUDY_DIR"

if [ ! -f "$STUDY_DIR/summary_results.csv" ]; then
    echo "The full study has not completed yet! (summary_results.csv missing)"
    echo "Check progress by running: tail -f studies/full_run_console.log"
    exit 1
fi

echo "==================================="
echo "1. Generating publication figures..."
echo "==================================="
.venv/bin/python plot_comparison_figures.py --study-dir "$STUDY_DIR"

echo "================================="
echo "2. Updating LaTeX section..."
echo "================================="

# Extract median values using python
read ten1 tom1 winner1 <<< $(.venv/bin/python -c "import pandas as pd; df = pd.read_csv('$STUDY_DIR/summary_results.csv'); s = df[df['experiment']=='exp1_p_sweep']; t=s['tenor_abs_err_p'].median(); o=s['tomchuk_abs_err_p_pdi'].median(); w='TENOR' if t < o*0.9 else 'Tomchuk'; print(f'{t:.3f} {o:.3f} \\\\{w}{{}}')")
read ten2 tom2 winner2 <<< $(.venv/bin/python -c "import pandas as pd; df = pd.read_csv('$STUDY_DIR/summary_results.csv'); s = df[df['experiment']=='exp2_flux']; t=s['tenor_abs_err_p'].median(); o=s['tomchuk_abs_err_p_pdi'].median(); w='TENOR' if t < o*0.9 else 'Tomchuk'; print(f'{t:.3f} {o:.3f} \\\\{w}{{}}')")
read ten3 tom3 winner3 <<< $(.venv/bin/python -c "import pandas as pd; df = pd.read_csv('$STUDY_DIR/summary_results.csv'); s = df[df['experiment']=='exp3_smearing']; t=s['tenor_abs_err_p'].median(); o=s['tomchuk_abs_err_p_pdi'].median(); w='TENOR' if t < o*0.9 else 'Tomchuk'; print(f'{t:.3f} {o:.3f} \\\\{w}{{}}')")
read ten4 tom4 winner4 <<< $(.venv/bin/python -c "import pandas as pd; df = pd.read_csv('$STUDY_DIR/summary_results.csv'); s = df[df['experiment']=='exp4_distribution']; t=s['tenor_abs_err_p'].median(); o=s['tomchuk_abs_err_p_pdi'].median(); w='TENOR' if t < o*0.9 else 'Tomchuk'; print(f'{t:.3f} {o:.3f} \\\\{w}{{}}')")
read ten5 tom5 winner5 <<< $(.venv/bin/python -c "import pandas as pd; df = pd.read_csv('$STUDY_DIR/summary_results.csv'); s = df[df['experiment']=='exp5_anisotropy']; t=s['tenor_abs_err_p'].median(); o=s['tomchuk_abs_err_p_pdi'].median(); w='TENOR' if t < o*0.9 else 'Tomchuk'; print(f'{t:.3f} {o:.3f} \\\\{w}{{}}')")

# Remove the fast-mode note and update the table in comparison_section.tex using perl
perl -i -0pe 's/%% NUMERICAL RESULTS: Filled from fast-mode study.*?%%\n%% Compile with/%% NUMERICAL RESULTS: Filled from FULL study (N=10 replicates).\n%%\n%% Compile with/s' comparison_section.tex

perl -i -0pe 's/\\caption\{Quantitative summary of the fast-mode comparative study \\(N=3 replicates per.*?\\label\{tab:comparison_summary\}/\\caption\{Quantitative comparison summary of the full study ($N=10$ replicates per condition). Values are median absolute errors $|\\precov - \\ptrue|$ pooled over all conditions within each experiment.\}\n\\label\{tab:comparison_summary\}/s' comparison_section.tex

# Update the table rows
perl -i -pe "s/\\\$p\\\$-sweep \\(all \\\$p \\\\in \\[0\\.1,0\\.6\\]\\)\\s+&.*?\\\\\\\\/\\\$p\\\$-sweep (all \\\$p \\\\in [0.1,0.6])   & $ten1 & $tom1 & $winner1 \\\\\\\\/" comparison_section.tex
perl -i -pe "s/Flux sensitivity \\(\\\$10\^7\\\$--\\\$10\^9\\\$\\)\\s+&.*?\\\\\\\\/Flux sensitivity (\\\$10^7\\\$--\\\$10^9\\\$)   & $ten2 & $tom2 & $winner2 \\\\\\\\/" comparison_section.tex
perl -i -pe "s/Smearing \\(\\\$\\sigma = 1\\\$--\\\$8\\\\,px\\)\\s+&.*?\\\\\\\\/Smearing (\\\$\\sigma = 1\\\$--\\\$8\\\\,px\\)    & $ten3 & $tom3 & $winner3 \\\\\\\\/" comparison_section.tex
perl -i -pe "s/Distribution shape \\(correct family\\)\\s+&.*?\\\\\\\\/Distribution shape (correct family) & $ten4 & $tom4 & $winner4 \\\\\\\\/" comparison_section.tex
perl -i -pe "s/PSF anisotropy\\s+&.*?\\\\\\\\/PSF anisotropy                      & $ten5 & $tom5 & $winner5 \\\\\\\\/" comparison_section.tex

# Copy to the study dir
cp comparison_section.tex "$STUDY_DIR/"
echo "LaTeX updated and copied to study directory."

echo "==================================="
echo "3. Compiling PDF..."
echo "==================================="
cd "$STUDY_DIR"
pdflatex -interaction=nonstopmode comparison_section.tex > /dev/null 2>&1
biber comparison_section > /dev/null 2>&1
pdflatex -interaction=nonstopmode comparison_section.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode comparison_section.tex > /dev/null 2>&1

if [ -f "comparison_section.pdf" ]; then
    echo "==================================="
    echo "DONE! PDF created at $STUDY_DIR/comparison_section.pdf"
    echo "You can copy it to your Desktop with:"
    echo "cp \"$STUDY_DIR/comparison_section.pdf\" ~/Desktop/"
else
    echo "Warning: compilation failed. Check $STUDY_DIR for logs."
fi
