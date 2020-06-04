#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include<unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <fstream>
using namespace cv;
using namespace std;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound, double higherBound) {
	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}
	#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv)
{
	// IO operation

	const char* keys =
		{
			"{ f  | vidFile      | ex2.avi | filename of video }"
			"{ x  | xFlowFile    | flow_x | filename of flow x component }"
			"{ y  | yFlowFile    | flow_y | filename of flow x component }"
			"{ i  | imgFile      | flow_i | filename of flow image}"
			"{ b  | bound | 15 | specify the maximum of optical flow}"
		};

	CommandLineParser cmd(argc, argv, keys);
	// string vidFile = cmd.get<string>("vidFile");
	// string xFlowFile = cmd.get<string>("xFlowFile");
	// string yFlowFile = cmd.get<string>("yFlowFile");
	// string imgFile = cmd.get<string>("imgFile");

	ifstream fin("/data/public/UCF101_Action_Recognition_Data_Set/UCF101_Action_detection_splits/train_rgb.txt");
	 if(!fin) { //打开失败
        cout << "error opening source file." << endl;
        return 0;
    }
	//string vidFile = "test.avi";
	string path="/data/public/UCF101_Action_Recognition_Data_Set/UCF-101/";
	string path_write="/data/ZBLiang/dense_data2/";
	string xFlowFile = "flow_x";
	string yFlowFile = "flow_y";
	string imgFile = "tmp/image";
	string videoName;
    while (getline(fin, videoName)){
    	char *str=new char[videoName.size()+1];
    	strcpy(str,videoName.c_str());
    	//cout<<str<<endl;
    	char *p;
    	p=strtok(str," ");
    	//cout<<p<<endl;
    	char fullname[20];
    	strcpy(fullname,p);
    	char *p1;
    	p1=strtok(str,"_");
    	int count=0;
    	char *videoname;
    	while(p1){
    		if(count==1) videoname=p1;
    		p1=strtok(NULL,"_");
    		count++;
    	}
    	//string fullName=p;
    	//string viName=videoname;
    	//cout<<p<<endl;
    	string vidFile=path+videoname+"/"+fullname+".avi";
	
		int bound = 20;

		cout<<vidFile<<endl;
		VideoCapture capture(vidFile);
		if(!capture.isOpened()) {
			printf("Could not initialize capturing..\n");
			return -1;
		}
		delete[] str;
		int frame_num = 0;
		Mat image, prev_image, prev_grey, grey, frame, flow, cflow;

		while(true) {
			capture >> frame;
			if(frame.empty())
				break;
	      
			if(frame_num == 0) {
				image.create(frame.size(), CV_8UC3);
				grey.create(frame.size(), CV_8UC1);
				prev_image.create(frame.size(), CV_8UC3);
				prev_grey.create(frame.size(), CV_8UC1);

				frame.copyTo(prev_image);
				cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

				frame_num++;
				continue;
			}

			frame.copyTo(image);
			cvtColor(image, grey, CV_BGR2GRAY);

			// calcOpticalFlowFarneback(prev_grey,grey,flow,0.5, 3, 15, 3, 5, 1.2, 0 );
	    calcOpticalFlowFarneback(prev_grey, grey, flow, 0.702, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
			
			// prev_image.copyTo(cflow);
			// drawOptFlowMap(flow, cflow, 12, 1.5, Scalar(0, 255, 0));

			Mat flows[2];
			split(flow,flows);
			Mat imgX(flows[0].size(),CV_8UC1);
			Mat imgY(flows[0].size(),CV_8UC1);
			convertFlowToImage(flows[0],flows[1], imgX, imgY, -bound, bound);
			char tmp[20];
			//CreateDirectory(path_write+fullName,NULL)
			if (access((path_write+fullname).c_str(), 0) == -1){
				mkdir((path_write+fullname).c_str(),0777);
			}
			sprintf(tmp,"_%05d.jpg",int(frame_num));
			imwrite(path_write+fullname+'/'+xFlowFile + tmp,imgX);
			imwrite(path_write+fullname+'/'+yFlowFile + tmp,imgY);
			//imwrite(imgFile + tmp, image);

			std::swap(prev_grey, grey);
			std::swap(prev_image, image);
			frame_num = frame_num + 1;
		}
	}
	return 0;
}
