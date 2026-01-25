# sdp observed:
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
