% 25_convert_colet_mat_to_tables.m
% Конвертация / инспекция COLET через MATLAB.
%
% Почему MATLAB:
%   COLET хранится как MATLAB v7.3/HDF5 с object references.
%   В Python/h5py разыменование ссылок работает слишком медленно.
%   MATLAB читает такие структуры нативно и должен быть основным способом конвертации.
%
% Как запускать из корня проекта:
%   matlab -batch "run('tools/25_convert_colet_mat_to_tables.m')"
%
% Что делает первая версия:
%   1. Загружает Data из COLET_v3/data_v3.mat.
%   2. Проверяет Data.subject_info и Data.task.
%   3. Для первых MAX_SUBJECTS и MAX_TASKS собирает структуру annotation/blinks/gaze/pupil.
%   4. Пытается создать компактные таблицы:
%        reports/wearable_pm_alignment/colet_matlab_structure_inventory.csv
%        reports/wearable_pm_alignment/colet_matlab_annotation_probe.csv
%   5. Не пытается сразу сохранять полные gaze/pupil/blinks.
%
% После успешного probe можно будет расширить этот скрипт до полного экспорта features.

clear; clc;

PROJECT_ROOT = 'D:\PycharmProjects\eeg-cognitive-state-nir';
COLET_VERSION = 'COLET_v3';
MAT_FILE = fullfile(PROJECT_ROOT, 'data', 'external', 'COLET', COLET_VERSION, 'data_v3.mat');
OUT_DIR = fullfile(PROJECT_ROOT, 'reports', 'wearable_pm_alignment');

MAX_SUBJECTS = 3;
MAX_TASKS = 4;
SAVE_FULL_DEBUG_MAT = false;

if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

fprintf('============================================================\n');
fprintf('COLET MATLAB probe/conversion\n');
fprintf('============================================================\n');
fprintf('MAT file: %s\n', MAT_FILE);
fprintf('Output dir: %s\n', OUT_DIR);

if ~exist(MAT_FILE, 'file')
    error('MAT file not found: %s', MAT_FILE);
end

fprintf('\nMAT file info:\n');
info = whos('-file', MAT_FILE);
disp(struct2table(info));

fprintf('\nLoading Data. This may take time for the first run...\n');
tLoad = tic;
S = load(MAT_FILE, 'Data');
loadSec = toc(tLoad);
fprintf('Loaded Data in %.2f sec\n', loadSec);

Data = S.Data;
clear S;

fprintf('\nTop-level Data fields:\n');
disp(fieldnames(Data));

nSubjects = numel(Data.task);
fprintf('Subjects in Data.task: %d\n', nSubjects);

if isfield(Data, 'subject_info')
    fprintf('Subjects in Data.subject_info: %d\n', numel(Data.subject_info));
end

maxSubjects = min(MAX_SUBJECTS, nSubjects);

structureRows = {};
annotationRows = {};

for s = 1:maxSubjects
    fprintf('\n------------------------------------------------------------\n');
    fprintf('Subject %d / %d\n', s, nSubjects);
    fprintf('------------------------------------------------------------\n');

    subjTask = Data.task{s};
    fprintf('task class: %s\n', class(subjTask));
    fprintf('task fields:\n');
    disp(fieldnames(subjTask));

    taskFields = fieldnames(subjTask);

    for f = 1:numel(taskFields)
        fieldName = taskFields{f};
        value = subjTask.(fieldName);

        structureRows(end+1, :) = {
            COLET_VERSION, s, NaN, fieldName, class(value), mat2str(size(value)), safeNumel(value), ''
        };

        fprintf('  %s: class=%s size=%s numel=%d\n', fieldName, class(value), mat2str(size(value)), safeNumel(value));
    end

    if isfield(subjTask, 'annotation')
        nTasks = numel(subjTask.annotation);
    else
        nTasks = 0;
    end

    maxTasks = min(MAX_TASKS, nTasks);

    for t = 1:maxTasks
        fprintf('\n  Task %d / %d\n', t, nTasks);

        for f = 1:numel(taskFields)
            fieldName = taskFields{f};
            fieldValue = subjTask.(fieldName);

            try
                if iscell(fieldValue)
                    leaf = fieldValue{t};
                elseif numel(fieldValue) >= t
                    leaf = fieldValue(t);
                else
                    continue;
                end
            catch ME
                structureRows(end+1, :) = {
                    COLET_VERSION, s, t, fieldName, 'READ_ERROR', '', NaN, ME.message
                };
                continue;
            end

            leafClass = class(leaf);
            leafSize = mat2str(size(leaf));
            leafNumel = safeNumel(leaf);

            childFields = '';
            if isstruct(leaf)
                childFields = strjoin(fieldnames(leaf), ';');
            elseif isobject(leaf)
                try
                    childFields = strjoin(properties(leaf), ';');
                catch
                    childFields = '';
                end
            end

            structureRows(end+1, :) = {
                COLET_VERSION, s, t, fieldName, leafClass, leafSize, leafNumel, childFields
            };

            fprintf('    %s{%d}: class=%s size=%s numel=%d fields=%s\n', ...
                fieldName, t, leafClass, leafSize, leafNumel, childFields);

            if strcmp(fieldName, 'annotation')
                annotationRows = appendAnnotationRow(annotationRows, COLET_VERSION, s, t, leaf);
            end
        end
    end
end

structureTable = cell2table(structureRows, 'VariableNames', {
    'version', 'subject_index', 'task_index', 'field_name', 'class_name', ...
    'size_str', 'numel_value', 'child_fields_or_note'
});

structureCsv = fullfile(OUT_DIR, 'colet_matlab_structure_inventory.csv');
writetable(structureTable, structureCsv);
fprintf('\nSaved structure inventory: %s\n', structureCsv);

if ~isempty(annotationRows)
    annotationTable = cell2table(annotationRows, 'VariableNames', {
        'version', 'subject_index', 'task_index', 'annotation_class', ...
        'annotation_size', 'annotation_numel', 'annotation_text', 'annotation_numeric_json'
    });

    annotationCsv = fullfile(OUT_DIR, 'colet_matlab_annotation_probe.csv');
    writetable(annotationTable, annotationCsv);
    fprintf('Saved annotation probe: %s\n', annotationCsv);
else
    fprintf('No annotation rows collected.\n');
end

if SAVE_FULL_DEBUG_MAT
    debugMat = fullfile(OUT_DIR, 'colet_debug_loaded_subset.mat');
    save(debugMat, 'structureTable', 'annotationTable', '-v7.3');
    fprintf('Saved debug MAT: %s\n', debugMat);
end

fprintf('\nDone.\n');


function n = safeNumel(x)
    try
        n = numel(x);
    catch
        n = NaN;
    end
end


function rows = appendAnnotationRow(rows, version, subjectIndex, taskIndex, annotation)
    annotationClass = class(annotation);
    annotationSize = mat2str(size(annotation));
    annotationNumel = safeNumel(annotation);
    annotationText = '';
    annotationNumericJson = '';

    try
        if ischar(annotation)
            annotationText = annotation;
        elseif isstring(annotation)
            annotationText = char(annotation);
        elseif isnumeric(annotation) || islogical(annotation)
            annotationNumericJson = jsonencode(annotation);
        elseif iscell(annotation)
            % Пытаемся компактно представить ячейки.
            parts = {};
            numericParts = {};
            for i = 1:min(numel(annotation), 20)
                item = annotation{i};
                if ischar(item) || isstring(item)
                    parts{end+1} = char(item); %#ok<AGROW>
                elseif isnumeric(item) || islogical(item)
                    numericParts{end+1} = jsonencode(item); %#ok<AGROW>
                elseif isstruct(item)
                    parts{end+1} = ['struct:' strjoin(fieldnames(item), ';')]; %#ok<AGROW>
                else
                    parts{end+1} = class(item); %#ok<AGROW>
                end
            end
            annotationText = strjoin(parts, ' | ');
            annotationNumericJson = strjoin(numericParts, ' | ');
        elseif isstruct(annotation)
            fn = fieldnames(annotation);
            parts = {};
            numericParts = {};
            for k = 1:numel(fn)
                v = annotation.(fn{k});
                if ischar(v) || isstring(v)
                    parts{end+1} = [fn{k} '=' char(v)]; %#ok<AGROW>
                elseif isnumeric(v) || islogical(v)
                    numericParts{end+1} = [fn{k} '=' jsonencode(v)]; %#ok<AGROW>
                else
                    parts{end+1} = [fn{k} '=' class(v) ':' mat2str(size(v))]; %#ok<AGROW>
                end
            end
            annotationText = strjoin(parts, ' | ');
            annotationNumericJson = strjoin(numericParts, ' | ');
        else
            annotationText = class(annotation);
        end
    catch ME
        annotationText = ['ANNOTATION_PARSE_ERROR: ' ME.message];
    end

    rows(end+1, :) = {
        version, subjectIndex, taskIndex, annotationClass, ...
        annotationSize, annotationNumel, annotationText, annotationNumericJson
    };
end
