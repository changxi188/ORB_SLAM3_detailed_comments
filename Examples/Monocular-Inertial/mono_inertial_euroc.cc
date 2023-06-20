/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza. Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of
 * Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>

#include <System.h>
#include "ImuTypes.h"

using namespace std;

void LoadImages(const string& strImagePath, const string& strPathTimes, vector<string>& vstrImages,
                vector<double>& vTimeStamps);

void LoadIMU(const string& strImuPath, vector<double>& vTimeStamps, vector<cv::Point3f>& vAcc,
             vector<cv::Point3f>& vGyro);

double ttrack_tot = 0;

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::INFO, "./log/ORB_SLAM3.log");

    FLAGS_colorlogtostderr = true;
    FLAGS_alsologtostderr  = true;
    FLAGS_minloglevel      = 0;

    if (argc < 5)
    {
        LOG(ERROR) << endl
                   << "Usage: ./mono_inertial_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1 "
                      "path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N "
                      "path_to_times_file_N) "
                   << endl;
        return 1;
    }

    const int num_seq = (argc - 3) / 2;
    LOG(INFO) << "num_seq = " << num_seq << endl;
    bool   bFileName = (((argc - 3) % 2) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc - 1]);
        LOG(INFO) << "file name: " << file_name << endl;
    }

    // Load all sequences:
    int                          seq;
    vector<vector<string> >      vstrImageFilenames;
    vector<vector<double> >      vTimestampsCam;
    vector<vector<cv::Point3f> > vAcc, vGyro;
    vector<vector<double> >      vTimestampsImu;
    vector<int>                  nImages;
    vector<int>                  nImu;
    vector<int>                  first_imu(num_seq, 0);

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq < num_seq; seq++)
    {
        LOG(INFO) << "Loading images for sequence " << seq << "...";

        string pathSeq(argv[(2 * seq) + 3]);
        string pathTimeStamps(argv[(2 * seq) + 4]);

        string pathCam0 = pathSeq + "/mav0/cam0/data";
        string pathImu  = pathSeq + "/mav0/imu0/data.csv";

        LoadImages(pathCam0, pathTimeStamps, vstrImageFilenames[seq], vTimestampsCam[seq]);
        LOG(INFO) << "LOADED!" << endl;

        LOG(INFO) << "Loading IMU for sequence " << seq << "...";
        LOG(INFO) << "path IMU : " << pathImu;
        LoadIMU(pathImu, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        LOG(INFO) << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if ((nImages[seq] <= 0) || (nImu[seq] <= 0))
        {
            LOG(ERROR) << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first

        while (vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--;  // first imu measurement to be considered
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    LOG(INFO).precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true);
    float             imageScale = SLAM.GetImageScale();
    LOG(INFO) << "Constructed System";

    for (seq = 0; seq < num_seq; seq++)
    {
        // Main loop
        cv::Mat                       im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        for (int ni = 0; ni < nImages[seq]; ni++)
        {
            // Read image from file
            im = cv::imread(vstrImageFilenames[seq][ni], cv::IMREAD_UNCHANGED);  // CV_LOAD_IMAGE_UNCHANGED);

            double tframe = vTimestampsCam[seq][ni];

            if (im.empty())
            {
                LOG(ERROR) << endl << "Failed to load image at: " << vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

            if (imageScale != 1.f)
            {
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#endif
                int width  = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
                t_resize = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t_End_Resize -
                                                                                                  t_Start_Resize)
                               .count();
                SLAM.InsertResizeTime(t_resize);
#endif
            }

            // Load imu measurements from previous frame
            vImuMeas.clear();

            if (ni > 0)
            {
                // LOG(INFO) << "t_cam " << tframe << endl;

                while (vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x, vAcc[seq][first_imu[seq]].y,
                                                             vAcc[seq][first_imu[seq]].z, vGyro[seq][first_imu[seq]].x,
                                                             vGyro[seq][first_imu[seq]].y, vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }
            }

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            // Pass the image to the SLAM system
            // LOG(INFO) << "tframe = " << tframe << endl;
            SLAM.TrackMonocular(im, tframe, vImuMeas);  // TODO change to monocular_inertial

            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

#ifdef REGISTER_TIMES
            t_track =
                t_resize + std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            ttrack_tot += ttrack;
            // LOG(INFO) << "ttrack: " << ttrack << std::endl;

            vTimesTrack[ni] = ttrack;

            // Wait to load the next frame
            double T = 0;
            if (ni < nImages[seq] - 1)
            {
                T = vTimestampsCam[seq][ni + 1] - tframe;
            }
            else if (ni > 0)
            {
                T = tframe - vTimestampsCam[seq][ni - 1];
            }

            if (ttrack < T)
            {
                usleep((T - ttrack) * 1e6);  // 1e6
            }
        }

        if (seq < num_seq - 1)
        {
            LOG(WARNING) << "--------------------- Changing the dataset ------------------------- \n\n" << endl;

            SLAM.ChangeDataset();
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    if (bFileName)
    {
        const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
        const string f_file  = "f_" + string(argv[argc - 1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}

void LoadImages(const string& strImagePath, const string& strPathTimes, vector<string>& vstrImages,
                vector<double>& vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t / 1e9);
        }
    }
}

void LoadIMU(const string& strImuPath, vector<double>& vTimeStamps, vector<cv::Point3f>& vAcc,
             vector<cv::Point3f>& vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while (!fImu.eof())
    {
        string s;
        getline(fImu, s);
        if (s[0] == '#')
            continue;

        if (!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int    count = 0;
            while ((pos = s.find(',')) != string::npos)
            {
                item          = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item    = s.substr(0, pos);
            data[6] = stod(item);

            vTimeStamps.push_back(data[0] / 1e9);
            vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
            vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
        }
    }
}
