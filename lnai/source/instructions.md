# Instructions

Instructions for analysing experiment scripts and plotting related outputs.

1. In `source/baseline.py` there is to be found an RL sorting approach where the agent is trained to learn how to sort.
    - We need to create a textual description of the adopted methodology, describe the environment, reward function and experimental setup - how many timesteps, curriculum description etc.
    - Store the report at `reports/baseline.tex`.
2. In `source/lev_subproc.py` there is a similar approach which additionally introduces a levenshtein distance component in the reward function to ensure faithfulness to a particular coaching sorting algorithm.
    - We need to create a textual description of the adopted methodology, describe the environment, reward function and experimental setup - how many timesteps, curriculum description etc.
    - Store the report at `reports/levenshtein.tex`.
    - We also need a variant that favours bubble sort, i.e., by using action masking to keep only adjacent swaps of the form (i, i+1) instead of generic (i, j) swaps. Store this on `lev_subproc_bubble.py`.
3. Next, there is `source/cpab.py`, in which we define a different experimental setup, utilising both RL by reward shaping, as well as Behavioral Cloning. Relevant source files are also `source/cpab_main.py` which is just the main execution script, and `source/lnai/rl/` imports found in `source/cpab.py`.
    - As with the previous ones, we need a textual description of the adopted methodology, describe the environment, reward function and experimental setup - how many timesteps, curriculum description etc.
    - Store the report at `reports/cpab.tex`.
    - We also need a variant of the above methodology that favours bubble sort, e.g., by appropriate action masking, as with the corresponding baseline. Append _Bubble to any source files, functions and class variants you need to create and store them as separate files from their original ones, but in the same directory (e.g., the corresponding to `source/cpab.py` should be `source/cpab_Bubbly.py` etc).
4. We shall now refactor `paper/add_lit_rev.tex` in the following way: All papers should be reviewed in one section while relevance to the camera ready should be separated in a different section. Also, do not use APA style but numbered citations, which will also affect some expressions, e.g., "\[3\] claim that" is no longer syntactically valid and other expressions like "Authors in \[3\] claim that" etc.
5. We shall now the results of the above experiments, as found in various files. To begin with, baseline results are not included here, so:
    - We need a testing script that will load `source/baseline_20_10000000.zip` and test it. You can also use `source/baseline.py` for reference on testing and plotting, even import utilities from there. Testing should involve `max(100, 2 * n^2)` test cases per value of `n`.
    - Store any baseline testing results in a corresponding directory `source/results/baseline`.
    - For cpab, we are interested in files / models with n=20, 10000000 timesteps, success rate (sr) 0.75, bc 0.0, 0.4 and 0.8 and coaching policies either bubble (eb) or selection (es). Those are also indicated in the names of the file in the form of `max_n20_t10000000_sr0.75_eb_bc0.4` for example.
    - For cpab we need some plots on training-related quantities, like rolling mean episode reward (e.g., rolling over 500 episodes), and rolling mean episode length. Plots should be grouped by expert / coach type (bubble and selection) and each plot should include results for all three configurations of bc (0.0, 0.4, 0.8).
    - For cpab, with a similar grouping as before, we also need sorting efficacy and coach conformity plots, based on the models. Again, `source/cpab.py` already contains some relevant utilities.
    - Store any related results in `source/results/cpab`.
    - Also, compose a report discussing the above results placed in `reports/results.tex`. Bear in mind that those models have been designed to be as generic as possible, i.e., they could potentially accept any swap-based sorting expert just by changing the expert advice function, with no other changes on the designed model itself.
6. Regarding CPAB, we need to repeat the testing process and the relevant plots using the following models:
    - For bubble coaching:
        - `cpab_res_max_n20_t2000000_sr0.75_es_bc0.0_sr.zip` -- this is the shared baseline approach between bubble and selectionc coaches.
        - `cpab_res_max_n20_t2000000_sr0.75_eb_bc0.4_sr.zip`
        - `cpab_res_max_n20_t2000000_sr0.75_eb_bc0.8_sr.zip`
    - For selection coaching:
        - `cpab_res_max_n20_t2000000_sr0.75_es_bc0.0_sr.zip`
        - `cpab_res_max_n20_t1000000_sr0.75_es_bc0.4_sr.zip`
        - `cpab_res_max_n20_t1000000_sr0.75_es_bc0.8_sr.zip`
    Analyses should reutilize scripts already developed in step 5, since they avoid re-running tests when there do exist relevant result files.
        - Also, revisit and update reports that are affected, i.e., `source/results.tex` and `reports/cpab.tex`.

7. Next, we are going to populate the revised paper with the new results. To that end, I have copied the corresponding reports into `paper/main.tex`.
    - First read the paper as is - many parts are disconnected, since this is an extended and revised version of a previous paper.
    - The main idea for this paper is to compare a RL approach that is generic enough to accomodate any plausible swap-based sorting coach without any architectural changes with the symbolic approach presented in "Coaching How To Search". Thus, we have developed the CPAB (don't use this acronym, its just for internal use) architecture.
    - First, we need to make the related literature section flow more naturally. Remove any uses of the `\paragraph{}` macro from that section (lines 115-194) and rearrange text, if needed, to logically sort and present related literature.
    - Then, the RL related section (lines 198-270) should present the experimental setup regarding CPAB and then the corresponding results. Move any related plots as `.png` files to `paper/assets/images` and consider regenerating those as PDF instead of PNG.
        - Also, maybe add a tikz diagram showcasing the overal CPAB architecture. Add related tikz source to `paper/assets/tikz`, in the same format as with the rest files there.
    - Next, maybe consider rephrasing some parts in the symbolic sorting section (lines 274-422) to align text and tone with the rest, if needed.
    - Then, maybe drawing some inspiration from `paper/paraphrasing/camera_ready.tex` introduction, discussion and conclusions sections, try to draft first a discussion section, then a concluding and then an introductory section.
    - As I will revise the paper, use the latex `\ednote{}` macro to place any comments you would like me to review - they will appear as red margin notes for me to read. In cases where you have any dilemmas, use an `\ednote{}` so as to know that you encountered such a situation. Also, leave such comments in any other cases you might see fit, if needed.