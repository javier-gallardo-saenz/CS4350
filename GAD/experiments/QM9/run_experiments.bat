@echo off

REM Define the alpha values and their corresponding model names
SET "ALPHA_VALUES=0.5 0.0 -0.5"
SET "MODEL_NAMES=1 2 3"

SET "MODEL_SAVE_DIR=saved_models"

REM Create directory if it doesn't exist
IF NOT EXIST "%MODEL_SAVE_DIR%" mkdir "%MODEL_SAVE_DIR%"

SET "BASE_CMD=python -m main_QM9 --n_layers=6 --hid_dim=50 --atomic_emb=25 --dropout=0 --readout=mean --use_diffusion=True --diffusion_method=implicit --k=25 --aggregators mean sum max dir_der --scalers identity amplification attenuation --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net=simple --towers=5 --prop_idx 0 1 2 --factor 1 1 10 --num_epochs=200 --batch_size=48 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5 --patience=25 --operator=Hub_Laplacian --gamma_diff=0 --gamma_adv=0 --diffusion_operator=Laplacian --learn_diff=False --diff_alpha=0 --diff_gamma_adv=0 --diff_gamma_diff=0"

REM Initialize counters for alpha values and model names
SET "ALPHA_COUNTER=0"
SET "NAME_COUNTER=0"

REM Loop through alpha values
FOR %%a IN (%ALPHA_VALUES%) DO (
    REM Get the current model name using delayed expansion
    CALL :GetModelName %%NAME_COUNTER%%
    SET "CURRENT_MODEL_NAME=!TEMP_MODEL_NAME!"

    ECHO Running with alpha = %%a, saving as model %CURRENT_MODEL_NAME%.pth

    SET "MODEL_PATH=%MODEL_SAVE_DIR%\model_%CURRENT_MODEL_NAME%.pth"

    REM Double quotes around %MODEL_PATH% are crucial
    CALL %BASE_CMD% --alpha=%%a --model_path="%MODEL_PATH%"

    ECHO Finished alpha = %%a
    ECHO -------------------------------------

    REM Increment counters
    SET /A ALPHA_COUNTER+=1
    SET /A NAME_COUNTER+=1
)

ECHO All experiments finished!
GOTO :EOF

REM Subroutine to get the model name at a specific index
:GetModelName
SET "TEMP_MODEL_NAME="
SET "CURRENT_INDEX=0"
FOR %%n IN (%MODEL_NAMES%) DO (
    IF !CURRENT_INDEX! EQU %1 (
        SET "TEMP_MODEL_NAME=%%n"
        GOTO :EOF
    )
    SET /A CURRENT_INDEX+=1
)
GOTO :EOF