#include<bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace chrono;

// 假设你已将所需的文件读取到 sizes 和 y_i 中，这里只是示例，实际情况下你需要读取数据文件。
const int C = 102400;  // KB
const int n = 300;     // 视频数量
const int timestep = 500;  // 时间步长

// 假设这些数据已经加载
vector<int> sizes(n);     // 视频文件大小
vector<vector<double>> y_i(n, vector<double>(timestep));  // 播放量
vector<vector<double>> r_del(n, vector<double>(timestep));  // 播放量

int main() {
    // 假设已经加载了 video_sizes 和 Video300_Time500 数据
    // 这里使用了默认数据填充，实际应根据具体的数据读取方式来填充
    ifstream file1("./video_sizes.txt");
    for (int i = 0; i < n; ++i) 
        file1 >> sizes[i];
    file1.close();
    ifstream file2("./LSTNet_testY.txt");
    for (int i = 0; i < n; ++i) 
        for (int t = 0; t < timestep; ++t) 
            file2 >> y_i[i][t];
    file2.close();

    // 计算 t1, t2 和 E1, E2
    double SNR1 = 72.27, e1 = 100, Wl = 100000;
    double SNR2 = 28.83, e2 = 50, esen = 0.00015, est = 0.0734;

    vector<double> t1(n), t2(n), E1(n), E2(n);
    for (int i = 0; i < n; ++i) {
        t1[i] = sizes[i] * 8.0 / (100000 * log2(1 + pow(10, SNR1 / 10.0)));
        E1[i] = t1[i] * e1;

        t2[i] = sizes[i] * 8.0 / (20000 * log2(1 + pow(10, SNR2 / 10.0)));
        E2[i] = E1[i] + t2[i] * e2 + esen * sizes[i] * 8 + est;
    }

    // 计算 r_del
    for (int i = 0; i < n; ++i) {
        for (int t = 0; t < timestep; ++t) {
            r_del[i][t] = y_i[i][t] * (E2[i] - E1[i]);
        }
    }

    // 动态规划求解
    vector<int> id;
    vector<vector<int>> index;
    vector<double> max_rewards;
    vector<double> timecost;
    // for (int t = 0; t < timestep; ++t) {  // 假设只考虑一个时间步
    for (int t = 0; t < timestep; ++t) {  // 假设只考虑一个时间步
        auto start = high_resolution_clock::now(); //记录当前时间。
        vector<double> dp(C + 1, 0);  // DP 状态数组
        vector<vector<int>> decision(n + 1, vector<int>(C + 1, 0));  // 记录选择的视频

        cout << t << ": " << '\n';
        for (int i = 1; i <= n; ++i) {  // 遍历每个视频
            for (int c = C; c >= sizes[i - 1]; --c) {  // 遍历存储容量，从大到小
                if (dp[c - sizes[i - 1]] + r_del[i - 1][t] > dp[c]) {
                    dp[c] = dp[c - sizes[i - 1]] + r_del[i - 1][t];
                    decision[i][c] = 1;  // 选择存储该视频
                } else {
                    decision[i][c] = 0;  // 不存储
                }
            }
        }

        // 找到最优存储策略
        id.clear();
        int c = C;
        for (int i = n; i > 0; --i) {
            if (decision[i][c] == 1) {
                id.push_back(i - 1);
                c -= sizes[i - 1];
            }
        }

        auto stop = high_resolution_clock::now();  // 记录结束时间
        auto duration = duration_cast<milliseconds>(stop - start);  // 计算持续时间
        
        cout << "选中的视频索引: ";
        for (auto i : id)
            cout << i << " ";
        cout << '\n';
        cout << "最大收益: " << dp[C] << " 耗时: " << duration.count() / 1000.0 <<'\n';
        
        index.push_back(id);
        max_rewards.push_back(dp[C]);
        timecost.push_back(duration.count() / 1000.0);
    }

    // 保存文件
    // ofstream out_file1("./dp_index.txt");
    ofstream out_file2("./optdp_rewards.txt");
    // ofstream out_file3("./dp_times.txt");

    // for (auto i : index){
    //     for (auto j : i){
    //         out_file1 << j << " ";
    //     }
    //     out_file1<<'\n';
    // }
    // out_file1.close();

    for (double reward : max_rewards) {
        out_file2 << reward << "\n";
    }
    out_file2.close();

    // for (double t : timecost) {
    //     out_file3 << t << "\n";
    // }
    // out_file3.close();

    return 0;
}
