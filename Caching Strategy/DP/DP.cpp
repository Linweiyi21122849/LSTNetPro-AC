#include<bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace chrono;

// �������ѽ�������ļ���ȡ�� sizes �� y_i �У�����ֻ��ʾ����ʵ�����������Ҫ��ȡ�����ļ���
const int C = 102400;  // KB
const int n = 300;     // ��Ƶ����
const int timestep = 500;  // ʱ�䲽��

// ������Щ�����Ѿ�����
vector<int> sizes(n);     // ��Ƶ�ļ���С
vector<vector<double>> y_i(n, vector<double>(timestep));  // ������
vector<vector<double>> r_del(n, vector<double>(timestep));  // ������
vector<vector<double>> y_i_real(n, vector<double>(timestep));  // ������
vector<vector<double>> r_del_real(n, vector<double>(timestep));  // ������

int main() {
    // �����Ѿ������� video_sizes �� Video300_Time500 ����
    // ����ʹ����Ĭ��������䣬ʵ��Ӧ���ݾ�������ݶ�ȡ��ʽ�����
    ifstream file1("./video_sizes.txt");
    for (int i = 0; i < n; ++i) 
        file1 >> sizes[i];
    file1.close();

    ifstream file2("./LSTNet_predict.txt");
    ifstream file3("./LSTNet_testY.txt");
    for (int i = 0; i < n; ++i) 
        for (int t = 0; t < timestep; ++t){
            file2 >> y_i[i][t];
            file3 >> y_i_real[i][t];
        }
    file2.close();
    file3.close();

    // ���� t1, t2 �� E1, E2
    double SNR1 = 72.27, e1 = 100, Wl = 100000;
    double SNR2 = 28.83, e2 = 50, esen = 0.00015, est = 0.0734;

    vector<double> t1(n), t2(n), E1(n), E2(n);
    for (int i = 0; i < n; ++i) {
        t1[i] = sizes[i] * 8.0 / (100000 * log2(1 + pow(10, SNR1 / 10.0)));
        E1[i] = t1[i] * e1;

        t2[i] = sizes[i] * 8.0 / (20000 * log2(1 + pow(10, SNR2 / 10.0)));
        E2[i] = E1[i] + t2[i] * e2 + esen * sizes[i] * 8 + est;
    }

    for (int i = 0; i < n; ++i) {
        for (int t = 0; t < timestep; ++t) {
            r_del[i][t] = y_i[i][t] * (E2[i] - E1[i]);
            r_del_real[i][t] = y_i_real[i][t] * (E2[i] - E1[i]);
        }
    }

    // ��̬�滮���
    vector<int> id;
    vector<vector<int>> index;
    vector<double> max_rewards;
    vector<double> hitrate;
    vector<double> max_rewards_real;
    vector<double> timecost;
    // for (int t = 0; t < timestep; ++t) {  // ����ֻ����һ��ʱ�䲽
    for (int t = 0; t < timestep; ++t) {  // ����ֻ����һ��ʱ�䲽
        auto start = high_resolution_clock::now(); //��¼��ǰʱ�䡣
        vector<double> dp(C + 1, 0);  // DP ״̬����
        vector<double> dp_real(C + 1, 0);  // DP ״̬����
        vector<vector<int>> decision(n + 1, vector<int>(C + 1, 0));  // ��¼ѡ�����Ƶ

        cout << t << ": " << '\n';
        for (int i = 1; i <= n; ++i) {  // ����ÿ����Ƶ
            for (int c = C; c >= sizes[i - 1]; --c) {  // �����洢�������Ӵ�С
                if (dp[c - sizes[i - 1]] + r_del[i - 1][t] > dp[c]) {
                    dp[c] = dp[c - sizes[i - 1]] + r_del[i - 1][t];
                    dp_real[c] = dp_real[c - sizes[i - 1]] + r_del_real[i - 1][t];
                    decision[i][c] = 1;  // ѡ��洢����Ƶ
                } else {
                    decision[i][c] = 0;  // ���洢
                }
            }
        }

        // �ҵ����Ŵ洢����
        id.clear();
        int c = C;
        for (int i = n; i > 0; --i) {
            if (decision[i][c] == 1) {
                id.push_back(i - 1);
                c -= sizes[i - 1];
            }
        }

        auto stop = high_resolution_clock::now();  // ��¼����ʱ��
        auto duration = duration_cast<milliseconds>(stop - start);  // �������ʱ��
        
        cout << "ѡ�е���Ƶ����: ";
        cout << "��ǰ��ʱ��" << t <<'\n';
        double sum = 0;
        for (auto i : id){
            cout << i << " ";
            sum += y_i_real[i][t];
        }
        cout << '\n';
        cout << "Ԥ���������: " << dp[C] << " ����������: " << sum << " ��ʱ: " << duration.count() / 1000.0 <<'\n';
        
        index.push_back(id);
        max_rewards.push_back(dp[C]);
        hitrate.push_back(sum);
        max_rewards_real.push_back(dp_real[C]);
        timecost.push_back(duration.count() / 1000.0);
    }

    // �����ļ�
    ofstream out_file1("./dp_index.txt");
    ofstream out_file2("./dp_rewards.txt");
    ofstream out_file3("./dp_times.txt");
    ofstream out_file4("./dp_rewards_real.txt");
    ofstream out_file5("./dp_hitrate.txt");

    for (auto i : index){
        for (auto j : i){
            out_file1 << j << " ";
        }
        out_file1<<'\n';
    }
    out_file1.close();

    for (double reward : max_rewards) {
        out_file2 << reward << "\n";
    }
    out_file2.close();

    for (double t : timecost) {
        out_file3 << t << "\n";
    }
    out_file3.close();

    for (double reward : max_rewards_real) {
        out_file4 << reward << "\n";
    }
    out_file4.close();

    for (double h : hitrate) {
        out_file5 << h << "\n";
    }
    out_file5.close();

    return 0;
}
