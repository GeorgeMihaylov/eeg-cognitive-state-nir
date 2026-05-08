# PM sampling statistics report

## Record processing status

| status   |   count |
|:---------|--------:|
| ok       |       5 |

## Records by source

| source   |   records |
|:---------|----------:|
| gpn_data |         5 |

## Metric-level recommendation

| source   |   pm_interval_median_s |   pm_interval_p75_s |   pm_interval_p90_s |   pm_interval_p95_s |   recommended_window_conservative_s |   recommended_window_balanced_s |   recommended_window_fast_s |
|:---------|-----------------------:|--------------------:|--------------------:|--------------------:|------------------------------------:|--------------------------------:|----------------------------:|
| gpn_data |                9.99442 |             9.99442 |             9.99442 |             9.99442 |                                9.99 |                            9.99 |                        9.99 |
| all      |                9.99442 |             9.99442 |             9.99442 |             9.99442 |                                9.99 |                            9.99 |                        9.99 |

## Aggregated PM interval statistics

| source   | metric                 | metric_type   |   records |   records_with_valid |   valid_count_median_per_record |   interval_median_s_median_across_records |   interval_p75_s_median_across_records |   interval_p90_s_median_across_records |   interval_p95_s_median_across_records |   value_mean_across_records |   active_ratio_mean |
|:---------|:-----------------------|:--------------|----------:|---------------------:|--------------------------------:|------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------:|--------------------:|
| gpn_data | PM.Attention.IsActive  | active        |         5 |                    3 |                               2 |                                 579.677   |                              794.557   |                              923.485   |                              966.461   |                    0.733333 |            0.733333 |
| gpn_data | PM.Engagement.IsActive | active        |         5 |                    2 |                               1 |                                  19.9888  |                              369.794   |                              601.664   |                              693.113   |                    0.8      |            0.8      |
| gpn_data | PM.Excitement.IsActive | active        |         5 |                    2 |                               1 |                                 314.824   |                              623.402   |                              795.056   |                              852.274   |                    0.853333 |            0.853333 |
| gpn_data | PM.Focus.IsActive      | active        |         5 |                    3 |                               2 |                                 559.688   |                              579.677   |                              837.533   |                              923.485   |                    0.733333 |            0.733333 |
| gpn_data | PM.Interest.IsActive   | active        |         5 |                    2 |                               1 |                                 444.752   |                              574.679   |                              700.609   |                              742.586   |                    0.8      |            0.8      |
| gpn_data | PM.Relaxation.IsActive | active        |         5 |                    3 |                               3 |                                  19.9888  |                              454.746   |                              751.581   |                              880.509   |                    0.733333 |            0.733333 |
| gpn_data | PM.Stress.IsActive     | active        |         5 |                    3 |                               2 |                                 559.688   |                              579.677   |                              837.533   |                              923.485   |                    0.733333 |            0.733333 |
| gpn_data | PM.Attention.Scaled    | scaled        |         5 |                    5 |                             185 |                                   9.99442 |                                9.99442 |                                9.99442 |                                9.99442 |                    0.427783 |          nan        |
| gpn_data | PM.Engagement.Scaled   | scaled        |         5 |                    5 |                             185 |                                   9.99442 |                                9.99442 |                                9.99442 |                                9.99442 |                    0.607733 |          nan        |
| gpn_data | PM.Excitement.Scaled   | scaled        |         5 |                    5 |                             199 |                                   9.99442 |                                9.99442 |                                9.99442 |                                9.99442 |                    0.4653   |          nan        |
| gpn_data | PM.Focus.Scaled        | scaled        |         5 |                    5 |                             185 |                                   9.99442 |                                9.99442 |                                9.99442 |                                9.99442 |                    0.503969 |          nan        |
| gpn_data | PM.Interest.Scaled     | scaled        |         5 |                    5 |                             185 |                                   9.99442 |                                9.99442 |                                9.99442 |                                9.99442 |                    0.494498 |          nan        |
| gpn_data | PM.Relaxation.Scaled   | scaled        |         5 |                    5 |                             185 |                                   9.99442 |                                9.99442 |                                9.99442 |                                9.99442 |                    0.357694 |          nan        |
| gpn_data | PM.Stress.Scaled       | scaled        |         5 |                    5 |                             185 |                                   9.99442 |                                9.99442 |                                9.99442 |                                9.99442 |                    0.46381  |          nan        |

## Per-record PM statistics preview

| source   | subject_id   | day   | metric                 | metric_type   |   valid_count |   interval_median_s |   interval_p90_s |   interval_p95_s |   value_mean |   active_ratio |
|:---------|:-------------|:------|:-----------------------|:--------------|--------------:|--------------------:|-----------------:|-----------------:|-------------:|---------------:|
| gpn_data | 0012905a     | day1  | PM.Attention.Scaled    | scaled        |           159 |             9.99442 |          9.99443 |          9.99443 |     0.420115 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Engagement.Scaled   | scaled        |           163 |             9.99442 |          9.99443 |          9.99443 |     0.585628 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Excitement.Scaled   | scaled        |           199 |             9.99442 |          9.99443 |          9.99443 |     0.617943 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Stress.Scaled       | scaled        |           160 |             9.99442 |          9.99443 |          9.99443 |     0.488619 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Relaxation.Scaled   | scaled        |           161 |             9.99442 |          9.99443 |          9.99443 |     0.344051 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Interest.Scaled     | scaled        |           161 |             9.99442 |          9.99443 |          9.99443 |     0.48657  |     nan        |
| gpn_data | 0012905a     | day1  | PM.Focus.Scaled        | scaled        |           160 |             9.99442 |          9.99443 |          9.99443 |     0.589704 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Attention.IsActive  | active        |             4 |           579.677   |        923.485   |        966.461   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Engagement.IsActive | active        |             8 |            19.9888  |        751.581   |        880.509   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Excitement.IsActive | active        |             5 |           344.808   |       1085.39    |       1172.35    |     0.6      |       0.6      |
| gpn_data | 0012905a     | day1  | PM.Stress.IsActive     | active        |             6 |           329.816   |        837.533   |        923.485   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Relaxation.IsActive | active        |             8 |            19.9888  |        751.581   |        880.509   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Interest.IsActive   | active        |             6 |           329.816   |        841.531   |        925.484   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Focus.IsActive      | active        |             6 |           329.816   |        837.533   |        923.485   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Attention.Scaled    | scaled        |            56 |             9.99442 |          9.99442 |          9.99442 |     0.362441 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Engagement.Scaled   | scaled        |            58 |             9.99442 |          9.99442 |          9.99442 |     0.565273 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Excitement.Scaled   | scaled        |            61 |             9.99442 |          9.99442 |          9.99442 |     0.808632 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Stress.Scaled       | scaled        |            56 |             9.99442 |          9.99442 |          9.99442 |     0.472605 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Relaxation.Scaled   | scaled        |            58 |             9.99442 |          9.99442 |          9.99442 |     0.293222 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Interest.Scaled     | scaled        |            56 |             9.99442 |          9.99442 |          9.99442 |     0.410179 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Focus.Scaled        | scaled        |            56 |             9.99442 |          9.99442 |          9.99442 |     0.759967 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Attention.IsActive  | active        |             2 |           559.688   |        559.688   |        559.688   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Engagement.IsActive | active        |             4 |            19.9888  |        451.748   |        505.718   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Excitement.IsActive | active        |             3 |           284.841   |        504.718   |        532.203   |     0.666667 |       0.666667 |
| gpn_data | 0012905a     | day1  | PM.Stress.IsActive     | active        |             2 |           559.688   |        559.688   |        559.688   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Relaxation.IsActive | active        |             4 |            19.9888  |        451.748   |        505.718   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Interest.IsActive   | active        |             2 |           559.688   |        559.688   |        559.688   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Focus.IsActive      | active        |             2 |           559.688   |        559.688   |        559.688   |     0.5      |       0.5      |
| gpn_data | 0012905a     | day1  | PM.Attention.Scaled    | scaled        |           360 |             9.99442 |          9.99442 |          9.99442 |     0.445857 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Engagement.Scaled   | scaled        |           361 |             9.99442 |          9.99442 |          9.99442 |     0.558373 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Excitement.Scaled   | scaled        |           361 |             9.99442 |          9.99442 |          9.99442 |     0.245053 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Stress.Scaled       | scaled        |           360 |             9.99442 |          9.99442 |          9.99442 |     0.451677 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Relaxation.Scaled   | scaled        |           358 |             9.99442 |          9.99442 |          9.99442 |     0.361916 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Interest.Scaled     | scaled        |           360 |             9.99442 |          9.99442 |          9.99442 |     0.520611 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Focus.Scaled        | scaled        |           360 |             9.99442 |          9.99442 |          9.99442 |     0.367896 |     nan        |
| gpn_data | 0012905a     | day1  | PM.Attention.IsActive  | active        |             3 |          1509.16    |       2708.49    |       2858.4     |     0.666667 |       0.666667 |
| gpn_data | 0012905a     | day1  | PM.Engagement.IsActive | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day1  | PM.Excitement.IsActive | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day1  | PM.Stress.IsActive     | active        |             3 |          1509.16    |       2708.49    |       2858.4     |     0.666667 |       0.666667 |
| gpn_data | 0012905a     | day1  | PM.Relaxation.IsActive | active        |             3 |          1509.16    |       2708.49    |       2858.4     |     0.666667 |       0.666667 |
| gpn_data | 0012905a     | day1  | PM.Interest.IsActive   | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day1  | PM.Focus.IsActive      | active        |             3 |          1509.16    |       2708.49    |       2858.4     |     0.666667 |       0.666667 |
| gpn_data | 0012905a     | day2  | PM.Attention.Scaled    | scaled        |           212 |             9.99442 |          9.99442 |          9.99442 |     0.446683 |     nan        |
| gpn_data | 0012905a     | day2  | PM.Engagement.Scaled   | scaled        |           212 |             9.99442 |          9.99442 |          9.99442 |     0.636386 |     nan        |
| gpn_data | 0012905a     | day2  | PM.Excitement.Scaled   | scaled        |           212 |             9.99442 |          9.99442 |          9.99442 |     0.386844 |     nan        |
| gpn_data | 0012905a     | day2  | PM.Stress.Scaled       | scaled        |           212 |             9.99442 |          9.99442 |          9.99442 |     0.458621 |     nan        |
| gpn_data | 0012905a     | day2  | PM.Relaxation.Scaled   | scaled        |           212 |             9.99442 |          9.99442 |          9.99442 |     0.414724 |     nan        |
| gpn_data | 0012905a     | day2  | PM.Interest.Scaled     | scaled        |           212 |             9.99442 |          9.99442 |          9.99442 |     0.535045 |     nan        |
| gpn_data | 0012905a     | day2  | PM.Focus.Scaled        | scaled        |           212 |             9.99442 |          9.99442 |          9.99442 |     0.432235 |     nan        |
| gpn_data | 0012905a     | day2  | PM.Attention.IsActive  | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day2  | PM.Engagement.IsActive | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day2  | PM.Excitement.IsActive | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day2  | PM.Stress.IsActive     | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day2  | PM.Relaxation.IsActive | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day2  | PM.Interest.IsActive   | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0012905a     | day2  | PM.Focus.IsActive      | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0110f12e     | day2  | PM.Attention.Scaled    | scaled        |           185 |             9.99443 |          9.99443 |          9.99443 |     0.463822 |     nan        |
| gpn_data | 0110f12e     | day2  | PM.Engagement.Scaled   | scaled        |           185 |             9.99443 |          9.99443 |          9.99443 |     0.693007 |     nan        |
| gpn_data | 0110f12e     | day2  | PM.Excitement.Scaled   | scaled        |           185 |             9.99443 |          9.99443 |          9.99443 |     0.268029 |     nan        |
| gpn_data | 0110f12e     | day2  | PM.Stress.Scaled       | scaled        |           185 |             9.99443 |          9.99443 |          9.99443 |     0.447529 |     nan        |
| gpn_data | 0110f12e     | day2  | PM.Relaxation.Scaled   | scaled        |           185 |             9.99443 |          9.99443 |          9.99443 |     0.374555 |     nan        |
| gpn_data | 0110f12e     | day2  | PM.Interest.Scaled     | scaled        |           185 |             9.99443 |          9.99443 |          9.99443 |     0.520087 |     nan        |
| gpn_data | 0110f12e     | day2  | PM.Focus.Scaled        | scaled        |           185 |             9.99443 |          9.99443 |          9.99443 |     0.370044 |     nan        |
| gpn_data | 0110f12e     | day2  | PM.Attention.IsActive  | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0110f12e     | day2  | PM.Engagement.IsActive | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0110f12e     | day2  | PM.Excitement.IsActive | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0110f12e     | day2  | PM.Stress.IsActive     | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0110f12e     | day2  | PM.Relaxation.IsActive | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0110f12e     | day2  | PM.Interest.IsActive   | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |
| gpn_data | 0110f12e     | day2  | PM.Focus.IsActive      | active        |             1 |           nan       |        nan       |        nan       |     1        |       1        |

## Interpretation

1. `interval_median_s` shows the typical interval between real PM updates.
2. `interval_p90_s` and `interval_p95_s` are more useful for selecting a robust window size.
3. If the chosen window is much shorter than the PM interval, many windows will have no PM target.
4. For a first baseline, use the balanced or conservative recommendation.
5. For a later real-time prototype, use shorter windows plus forward-fill/nearest PM assignment.