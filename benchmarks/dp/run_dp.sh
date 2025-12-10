# sdp observed:
# python main_dp.py --optimize --algorithm sdp --sim-start-date 1970-10-01 --sim-end-date 1995-09-30 --deficit-penalty-beta 3.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_train --starting-storage 450
# python main_dp.py --simulate --algorithm sdp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 3.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_train --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm sdp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 2.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_test --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm sdp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 3.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_test --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm sdp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 6.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_test --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm sdp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 9.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_test --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm sdp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 12.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_test --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm sdp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 20.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_test --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm sdp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 30.0 --inflow-file ../../data/folsom_daily.csv --lognormal-qstat q_stat_obs_test --starting-storage 466.1

# ddp observed:
python main_dp.py --optimize --simulate --algorithm ddp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 2.0 --inflow-file ../../data/folsom_daily.csv --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm ddp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 3.0 --inflow-file ../../data/folsom_daily.csv --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm ddp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 6.0 --inflow-file ../../data/folsom_daily.csv --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm ddp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 9.0 --inflow-file ../../data/folsom_daily.csv --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm ddp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 12.0 --inflow-file ../../data/folsom_daily.csv --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm ddp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 20.0 --inflow-file ../../data/folsom_daily.csv --starting-storage 466.1
python main_dp.py --optimize --simulate --algorithm ddp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 30.0 --inflow-file ../../data/folsom_daily.csv --starting-storage 466.1


# sdp paleo:
# python main_dp.py --optimize --algorithm sdp --sim-start-date 1899-10-01 --sim-end-date 1999-09-30 --deficit-penalty-beta 1.0 --inflow-file ../../data/folsom_paleo-train.csv --lognormal-qstat q_stat_paleo_train --starting-storage 450
# python main_dp.py --optimize --algorithm sdp --sim-start-date 1899-10-01 --sim-end-date 1999-09-30 --deficit-penalty-beta 2.0 --inflow-file ../../data/folsom_paleo-train.csv --lognormal-qstat q_stat_paleo_train --starting-storage 450
# python main_dp.py --optimize --algorithm sdp --sim-start-date 1899-10-01 --sim-end-date 1999-09-30 --deficit-penalty-beta 3.0 --inflow-file ../../data/folsom_paleo-train.csv --lognormal-qstat q_stat_paleo_train --starting-storage 450
# python main_dp.py --simulate --algorithm sdp --sim-start-date 1899-10-01 --sim-end-date 1999-09-30 --deficit-penalty-beta 1.0 --inflow-file ../../data/folsom_paleo_test.csv --lognormal-qstat q_stat_paleo_train --starting-storage 450
# python main_dp.py --simulate --algorithm sdp --sim-start-date 1899-10-01 --sim-end-date 1999-09-30 --deficit-penalty-beta 2.0 --inflow-file ../../data/folsom_paleo_test.csv --lognormal-qstat q_stat_paleo_train --starting-storage 450
# python main_dp.py --simulate --algorithm sdp --sim-start-date 1899-10-01 --sim-end-date 1999-09-30 --deficit-penalty-beta 3.0 --inflow-file ../../data/folsom_paleo_test.csv --lognormal-qstat q_stat_paleo_train --starting-storage 450

# sdp cc median:
# python main_dp.py --optimize --simulate --algorithm sdp --inflow-file ../../data/folsom-cc_median_50y_scen.csv --lognormal-qstat q_stat_cc --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 1.0 --starting-storage 450
# python main_dp.py --optimize --simulate --algorithm sdp --inflow-file ../../data/folsom-cc_median_50y_scen.csv --lognormal-qstat q_stat_cc --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 2.0 --starting-storage 450
# python main_dp.py --optimize --simulate --algorithm sdp --inflow-file ../../data/folsom-cc_median_50y_scen.csv --lognormal-qstat q_stat_cc --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 3.0 --starting-storage 450

# sdp cc driest:
# python main_dp.py --simulate --algorithm sdp --sdp_policy_dir ./output/median_50y_cc/ --inflow-file ../../data/folsom-cc_driest_50y_scen.csv --lognormal-qstat q_stat_cc --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 1.0 --starting-storage 450
# python main_dp.py --simulate --algorithm sdp --sdp_policy_dir ./output/median_50y_cc/ --inflow-file ../../data/folsom-cc_driest_50y_scen.csv --lognormal-qstat q_stat_cc --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 2.0 --starting-storage 450
# python main_dp.py --simulate --algorithm sdp --sdp_policy_dir ./output/median_50y_cc/ --inflow-file ../../data/folsom-cc_driest_50y_scen.csv --lognormal-qstat q_stat_cc --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 3.0 --starting-storage 450

# ddp observed:
# python main_dp.py --optimize --simulate --algorithm ddp --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 --deficit-penalty-beta 2.0 --starting-storage 450

# ddp cc median:
# python main_dp.py --optimize --simulate --algorithm ddp --inflow-file ../../data/folsom-cc_median_50y_scen.csv --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 1.0 --starting-storage 450
# python main_dp.py --optimize --simulate --algorithm ddp --inflow-file ../../data/folsom-cc_median_50y_scen.csv --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 2.0 --starting-storage 450
# python main_dp.py --optimize --simulate --algorithm ddp --inflow-file ../../data/folsom-cc_median_50y_scen.csv --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 3.0 --starting-storage 450

# ddp cc driest:
# python main_dp.py --optimize --simulate --algorithm ddp --inflow-file ../../data/folsom-cc_driest_50y_scen.csv --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 1.0 --starting-storage 450
# python main_dp.py --optimize --simulate --algorithm ddp --inflow-file ../../data/folsom-cc_driest_50y_scen.csv --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 2.0 --starting-storage 450
# python main_dp.py --optimize --simulate --algorithm ddp --inflow-file ../../data/folsom-cc_driest_50y_scen.csv --sim-start-date 1999-10-01 --sim-end-date 2099-09-30 --deficit-penalty-beta 3.0 --starting-storage 450

