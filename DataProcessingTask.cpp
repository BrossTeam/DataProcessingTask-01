// data_processing_task.cpp : Defines the entry point for the application.
//

#include "DataProcessingTask.h"
#include "ctranslate2/models/whisper.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <filesystem>


std::vector<std::vector<float>> read_csv_matrix(const char* file_name)
{
    // Open the CSV file
    std::ifstream file(file_name);

    // Initialize a vector to hold the rows
    std::vector<std::vector<float>> data;

    // Read the file line by line
    std::string line;
    while (std::getline(file, line))
    {
        std::vector<float> row;

        // Use a stringstream to parse the comma-separated values
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ','))
        {
            row.push_back(std::stof(value));
        }

        // Add the row to the data vector
        data.push_back(row);
    }
    return data;
}

/*
 * TODO: the task for you is to complete the following function
 *
 * input: a (80, 3000) metrix
 *
 * output: StorageView type
 *
 * hint: go to definition of StorageView
 */
 
ctranslate2::Shape  GetShape(std::vector<std::vector<std::vector<float>>>& array) {
    ctranslate2::Shape  shape;
    shape.push_back(array.size());
    shape.push_back(array[0].size());
    shape.push_back(array[0][0].size());
    return shape;
}


float* ascontiguousarray(std::vector<std::vector<std::vector<float>>>& array) {
    int dim1 = array.size();
    int dim2 = array[0].size();
    int dim3 = array[0][0].size();

    float* data = new float[dim1*dim2*dim3+100];
    unsigned long long count = 0;
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                data[count] = array[i][j][k];
                ++count;
            }
        }
    }
    return data;
}


ctranslate2::StorageView create_view_from_array(std::vector<std::vector<std::vector<float>>>& array) {
    // old python's c++ code,to pure c++ code
    //auto device = Device::CPU;
    auto device = ctranslate2::Device::CPU;


    /*  It is not a Python object, so the following operation is not required
    py::object interface_obj = py::getattr(array, "__array_interface__", py::none());
    if (interface_obj.is_none()) {
        interface_obj = py::getattr(array, "__cuda_array_interface__", py::none());
        if (interface_obj.is_none())
            throw std::invalid_argument("Object does not implement the array interface");
        device = Device::CUDA;
    }
    
    py::dict interface = interface_obj.cast<py::dict>();
    if (interface_obj.contains("strides") && !interface_obj["strides"].is_none())
        throw std::invalid_argument("StorageView does not support arrays with non contiguous memory");
    */


    //auto shape = interface["shape"].cast<Shape>();
    ctranslate2::Shape shape = GetShape(array);
    //auto dtype = typestr_to_dtype(interface["typestr"].cast<std::string>());
    auto dtype = ctranslate2::DataType::FLOAT32;
    //auto data = interface["data"].cast<py::tuple>();
    //auto ptr = data[0].cast<uintptr_t>();
    float* ptr = ascontiguousarray(array);
    //std::unique_ptr<float> auto_ptr(ptr);
    /*
    auto read_only = data[1].cast<bool>();
    if (read_only)
        throw std::invalid_argument("StorageView does not support read-only arrays");
    */
    //ctranslate2::StorageView view(dtype, device);
    ctranslate2::StorageView view(dtype, device);
    //view.view((void*)auto_ptr.get(), std::move(shape));
    view.view((void*)ptr, std::move(shape));
    return view;
}

ctranslate2::StorageView from_arry(std::vector<std::vector<std::vector<float>>>& array) {
    // old python code, to c++
    //StorageViewWrapper view = create_view_from_array(array);
    ctranslate2::StorageView view = create_view_from_array(array);
    //view.set_data_owner(array);  //Set the owner of the array object in Python, telling the program that the array object has the same lifecycle as the view.
    //return view;
    return view;
}


void Correct2Rectangle(std::vector<std::vector<float>>& segment) {
    unsigned long long row_count = segment.size();
    if (row_count <= 0) {
        return ;
    }

    unsigned long long col_count = 0;
    for (unsigned long long i = 0; i < row_count; ++i) {
        if (col_count < segment.at(i).size()) {
            col_count = segment.at(i).size();
        }
    }
    

    for (unsigned long long i = 0; i < row_count; ++i) {
        unsigned long long size = segment.at(i).size();
        for (unsigned long long j = size; j < col_count; ++j) {
            col_count = segment.at(i).size();
            segment[i][j]= 0.0;
        }
    }
}
ctranslate2::StorageView get_ctranslate2_storage(std::vector<std::vector<float>>& segment)
{
    Correct2Rectangle(segment); //Correct data to a regular rectangle


    //auto device = ctranslate2::Device::CPU;
    //auto dtype = ctranslate2::DataType::FLOAT32;
    //ctranslate2::StorageView view(dtype, device);
    
    
    // old python code, to c++
    //segment = segment.astype(np.float32)   //The variable type of C++is predefined and can only be float. So there's no need for this step    
    //segment = np.ascontiguousarray(segment) //
    //segment = np.expand_dims(segment, 0)
    
    
    std::vector<std::vector<std::vector<float>>> segment_3D;
    segment_3D.push_back(segment);
    //segment = ctranslate2.StorageView.from_array(segment)
    ctranslate2::StorageView view = from_arry(segment_3D);
    /////////////////////////////////////////////////////
    return view;
}


int main()
{
    // init and define options for whisper
    ctranslate2::models::WhisperOptions options;
    options.beam_size = 1;
    options.patience = 1;
    options.length_penalty = 1;
    options.max_length = 448;
    options.return_scores = true;
    options.return_no_speech_prob = true;
    options.suppress_blank = true;
    options.max_initial_timestamp_index = 50;

    // define prompts
    std::vector<std::vector<size_t>> prompts{ std::vector<size_t>{ 50257 } };

    // load the whisper model
    ctranslate2::models::Whisper whisper_model("./whisper-tiny.en-ct2", ctranslate2::Device::CPU);

    // load a 30 second data sample 
    auto segment = read_csv_matrix("./audio_test.csv");

    // load the vector<vector<float>> type into a ctranslate2::StorageView type
    auto features = get_ctranslate2_storage(segment);

    // perform inference
    auto result = whisper_model.generate(features, prompts, options);
    auto res = result[0].get();

    // print the result of the test
	std::cout << res.sequences[0][1] << std::endl;
    
    if (res.sequences[0][1] == "ĠHello") std::cout << "Good job, go collect your prize!";
    else std::cout << "Failed, sheesh gotta try again!";

	return 0;
}
