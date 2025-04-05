import os
import torch
import shutil
import pickle
import zipfile
import matplotlib
import numpy as np
import gradio as gr
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from pandas.plotting import andrews_curves
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset

###################################################### Preprocessing #####################################################################
def preprocess_dataframe(df, target_column=None, fill_method='mean', drop_na=True, sequence_length=32, test_size=0.2, batch_size = 128):
    """
    Loads a DataFrame from a file, preprocesses it, prepares it for LSTM data.
    If a target_column is provided, that column is used as the target (y).
    Otherwise, it prepares data for an autoencoder (no separate y).
    1. Loads file and checks for the target columns
    2. Drops any NaN rows and non numeric columns.
    3. Fills the NaN values with given method.
    4. After preprocessing, data is transformed to fit in lstm.

    Args:
        file_path (str): Path to the data file (e.g., CSV, Excel).
        target_column (str, optional): Name of the target column. If provided, use this as target. Otherwise, treats as autoencoder.  Defaults to None.
        fill_method (str, optional): Method for filling NaNs: 'mean', 'median', 'most_frequent', or 'constant'.
                                       Defaults to 'mean'. If 'constant', `fill_value` must be set.
        drop_na (bool, optional): Whether to drop rows with any NaN values. Defaults to True.
        sequence_length (int): The length of the sequence to create (e.g., number of features to treat as a sequence).
        test_size (float): The proportion of data to use for testing.

    Returns:
        tuple: (train_loader, test_loader, input_size) if no target_column.
               (train_loader, test_loader, input_size, target_column_name) if target_column provided
        A tuple containing:
            - train_loader (DataLoader): DataLoader for training data.
            - test_loader (DataLoader): DataLoader for test data.
            - input_size (int): Number of features.
            - target_column_name (str): The name of the target column only when there is target column.
    """
    # 1. Target Column Check
    target_col = None
    if target_column:
        if target_column in df.columns:
            target_col = target_column
            print(f"Target column '{target_column}' found.")
        else:
            target_column = None  # Reset target_column so we treat as autoencoder
    else:
        print("No target column specified. Treating as autoencoder.")

    #2. Drop Rows with NaNs before Fill
    if drop_na:
        print("Dropping rows with any NaN values...")
        df = df.dropna()


    # 3. Drop Non-Numeric Columns (Except Target)
    columns_to_drop = []
    for col in df.columns:
        if col != target_col and not pd.api.types.is_numeric_dtype(df[col]):
             columns_to_drop.append(col) #exclude the target column if target column is not numeric
    if columns_to_drop:
        print(f"Dropping non-numeric columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    else:
        print("No non-numeric columns found.")


    # 4. Handle Missing Values (Only in Numeric Columns After Dropping)
    numeric_cols = df.select_dtypes(include=np.number).columns #select numeric columns after non-numeric columsn removed
    if df[numeric_cols].isnull().any().any():  # Check if any NaN values exist (in numeric columns)
        print("Handling missing values...")
        if fill_method in ['mean', 'median', 'most_frequent', 'constant']:
            imputer = SimpleImputer(strategy=fill_method)

            if fill_method == 'constant':
                 imputer = SimpleImputer(strategy=fill_method, fill_value=0) #only with constant filling value must be provided

            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])  # Apply only to numeric columns

        else:
            raise ValueError("Invalid fill_method. Choose 'mean', 'median', 'most_frequent', or 'constant'.")

    # Droping NaN and inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if target_col:
        inputdf = df.drop(columns=[target_col])
        outputdf = df[target_col].apply(lambda x: 0 if x.lower() == 'benign' else 1)
        malinputdf = inputdf[outputdf == 1]
        beninputdf = inputdf[outputdf == 0]
        sample_size = min(len(beninputdf), len(malinputdf), 500)
        bensample = beninputdf.sample(n=sample_size, random_state=42)
        bensample['Label'] = 'Benign'
        malsample = malinputdf.sample(n=sample_size, random_state=42)
        malsample['Label'] = 'Malicious'
        sample = pd.concat([bensample, malsample])
        data = beninputdf.values
    else:
        inputdf = df
        sample_size = min(len(inputdf), 500)
        sample = df.sample(n=sample_size, random_state=42)
        data = inputdf.values

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    if target_col:
      X_train = data
      data = malinputdf.values
      data = scaler.transform(data)
      X_test = data
    else:
      X_train = data

    class TabularDatasetTest(Dataset):
      def __init__(self, data):
          self.data = data.clone().detach()

      def __len__(self):
          return len(self.data)

      def __getitem__(self, idx):
          return self.data[idx], self.data[idx]

    class TabularDatasetTrain(Dataset):
      def __init__(self, data, sequence_length):
          self.data = data.clone().detach()
          self.sequence_length = sequence_length

      def __len__(self):
          return len(self.data) - self.sequence_length + 1

      def __getitem__(self, idx):
          return self.data[idx:idx + self.sequence_length], self.data[idx:idx + self.sequence_length]

    if target_column:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        train_dataset = TabularDatasetTrain(X_train, sequence_length = sequence_length)
        test_dataset = TabularDatasetTest(X_test)
        train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_Dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return {
            'train_loader': train_DataLoader,
            'test_loader': test_Dataloader,
            'input_df': inputdf,
            'target_df': outputdf,
            'malinput_df': malinputdf,
            'beninput_df': beninputdf,
            'target_col': target_col,
            'scaler': scaler,
            'sample': sample
        }
    else:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        train_dataset = TabularDatasetTrain(X_train, sequence_length = sequence_length)
        train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        sample["Label"] = "dummy_class"
        return {
            'train_loader': train_DataLoader,
            'test_loader': None,
            'input_df': inputdf,
            'malinput_df': None,
            'beninput_df': None,
            'target_df': None,
            'target_col': None,
            'scaler': scaler,
            'sample': sample
        }
################################################## Model #############################################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bottleneck_size = int(input_size/2)

        self.isCuda = isCuda
        self.lstm1 = nn.LSTM(input_size, int(hidden_size/2), num_layers, batch_first=True, bidirectional = True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, self.bottleneck_size, num_layers, batch_first=True)


    def forward(self, inputs):
        intermediate_state, hidden = self.lstm1(inputs)#, (h0_1, c0_1))
        intermediate_state = self.relu(self.dropout(intermediate_state))
        encoded_input, hidden = self.lstm2(intermediate_state)#, (h0_2, c0_2))
        return encoded_input, intermediate_state

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bottleneck_size = int(output_size/2)

        self.isCuda = isCuda
        self.lstm2 = nn.LSTM(self.bottleneck_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(2*hidden_size, output_size, num_layers, batch_first=True)

    def forward(self, encoded_input, intermediate_state):
        encoded_input, hidden = self.lstm2(encoded_input)#, (h0_2, c0_2))
        inputs = torch.cat((self.dropout(encoded_input), intermediate_state), dim=2)
        inputs = self.relu(inputs)
        decoded_output, hidden = self.lstm1(inputs)#, (h0_1, c0_1))
        # print(f"output: {decoded_output}")
        return decoded_output

class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, isCuda="cuda" if torch.cuda.is_available() else "cpu"):
        super(LSTMAE, self).__init__()
        hidden_size = hidden_size if hidden_size%2==0 else hidden_size+1
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, isCuda)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the linear, LSTM, and convolutional layers
        using appropriate initialization schemes.
        """
        for m in self.modules():  # Iterate through all modules in the network
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        if 'ih' or 'hh' in name:
                            nn.init.xavier_uniform_(param.data)  # Input-to-hidden
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, input):
        encoded_input, intermediate_state = self.encoder(input)
        decoded_output = self.decoder(encoded_input, intermediate_state)
        return decoded_output



############################################## Andrews Curves ###########################################################################
def make_better_andrews_curves(df, class_column, colors=None, plot_title="Andrews Curves",
                               line_width=0.8, transparency=0.5,  sample_size=None, legend_loc='best',
                               custom_labels=None,  x_axis_ticks=None, x_axis_labels=None,
                               figsize=(10, 6), dpi=300, name = "andrews_curves"):
    """
    Generates an Andrews Curves plot with enhanced styling.

    Args:
        df: pandas DataFrame containing the data.
        class_column: Name of the column containing class labels.
        colors: List of colors to use for each class (e.g., ['blue', 'red']).  Defaults to matplotlib's defaults if None.
        plot_title: Title of the plot.
        line_width: Width of the lines.
        transparency: Alpha value (transparency) of the lines.
        sample_size: If an integer is provided, a random sample of the data will be used.  Useful for large datasets.
        legend_loc: Location of the legend (e.g., 'best', 'upper right', 'lower left').
        custom_labels: A dictionary mapping original class labels to more descriptive labels for the legend.
        x_axis_ticks: A list of tick positions for the x-axis. If None, default ticks are used.
        x_axis_labels: A list of labels for the x-axis ticks. Must be the same length as x_axis_ticks.
        figsize: Tuple specifying the figure size (width, height) in inches.
    """

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)  # Sample for faster plotting

    plt.figure(figsize=figsize)  # Set the figure size before plotting

    ax = andrews_curves(df, class_column, color=colors)  # Store the Axes object

    plt.title(plot_title, fontsize=16)
    plt.xlabel("t", fontsize=12)  # Added x-axis label
    plt.ylabel("f(t)", fontsize=12) # Added y-axis label

    for line in ax.get_lines():
        line.set_linewidth(line_width)
        line.set_alpha(transparency)

    # Customize Legend
    if custom_labels:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [custom_labels.get(label, label) for label in labels] # Use .get() to handle missing labels
        ax.legend(handles, new_labels, loc=legend_loc, fontsize=10)
    else:
        plt.legend(loc=legend_loc, fontsize=10)


    # Customize X-axis ticks and labels
    if x_axis_ticks:
        plt.xticks(x_axis_ticks, x_axis_labels)

    plt.grid(False)  # Add a grid
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(f"{name}.png", dpi=dpi)
################################################# Model Training ######################################################################
<<<<<<< HEAD
def train_model(model, train_loader, test_loader = None, learning_rate=0.001, epochs=10):
=======
def train_model(model, train_loader, test_loader = None, learning_rate=0.001, epochs=10, device = "cuda" if torch.cuda.is_available() else "cpu"):
>>>>>>> 0f1b551798e83326b79fb965a64927cbd7d89739
    criterion = nn.MSELoss()
    info = ""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_data = {}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_train_losses = []
        mse_losses = []
        for i,(inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            # l1_lambda = 0.001
            # l2_lambda = 0.0001
            # l1_norm = sum(p.abs().sum() for p in model.parameters())  # L1 norm
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())  # L2 norm
            loss = criterion(outputs, targets)# + l2_lambda * l2_norm + l1_lambda * l1_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == epochs-1:
              mse_loss = F.mse_loss(targets, outputs, reduction='none')
              mse_loss_per_data_point = mse_loss.mean(dim=-1)
              mse_losses.extend(mse_loss_per_data_point.tolist())
            epoch_train_losses.append(loss.item())
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        if test_loader and epoch%1==0:
          model.eval()
          test_loss = 0.0
          with torch.no_grad():
              for i,(inputs, targets) in enumerate(test_loader):
                  inputs = inputs.to(device)
                  targets = targets.to(device)
                  outputs = model(inputs.unsqueeze(1))
                  loss = criterion(outputs.squeeze(1), targets)
                  test_loss += loss.item()

          test_loss /= len(test_loader)
        else:
          test_loss = 0.0
        info += f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n"
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        train_loss_data[f'Epoch {epoch + 1}'] = epoch_train_losses
    train_loss_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in train_loss_data.items()]))
    return model, train_loss_df, mse_losses, info

#########################################################################################################################################
def detect_anomalies(csv_file, sample_choice="Custom Data", data_slicing_percentage=80, epochs=3, threshold_factor=1.0):
  images = []
  anomaly_summary = ""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if os.path.exists("Results"):
    shutil.rmtree("Results")
  os.mkdir("Results")
  if sample_choice == "Custom Data":
    anomaly_summary += f"[INFO] Loading Custom Dataset {data_slicing_percentage}%...\n"
    dataframe = pd.read_csv(csv_file.name).sample(frac=data_slicing_percentage/100, random_state=42).reset_index(drop=True)
    anomaly_summary += f"[INFO] Preprocessing Dataset...\n"
    if dataframe.get('Label') is not None:
        processed_data = preprocess_dataframe(dataframe, target_column="Label")
    else:
        processed_data = preprocess_dataframe(dataframe)
        anomaly_summary += f"[WARNING] No Label Column Found, Using Unsupervised Learning...\n"
    anomaly_summary += f"[INFO] Generating Andrews Curves...\n"
    make_better_andrews_curves(processed_data['sample'], 'Label',
                              colors=['Blue', 'Red'],
                              plot_title="Dataset Andrews Curves",
                              line_width=1.2,
                              transparency=0.7,
                              legend_loc='upper right',
                              figsize=(12, 7),
                              name = "Results/Dataset_andrews_curves")
    images.append("Results/Dataset_andrews_curves.png")
    model = LSTMAE(len(processed_data["input_df"].columns),128).to(device)
    model.to(device)
    anomaly_summary += f"[INFO] Training Model...\n"
    _, train_loss_df, mse_losses, info = train_model(model, processed_data['train_loader'], processed_data['test_loader'],epochs=epochs)
    anomaly_summary += info
    anomaly_summary += f"[INFO] Saving model, scaler, Dataset Used...\n"
    dataframe.to_csv('Results/Original_dataset.csv', columns=dataframe.columns, index=False)
    pickle.dump(processed_data['scaler'], open('Results/scaler.pkl', 'wb'))
    torch.save(model, 'Results/model.pth')
    anomaly_summary += f"[INFO] Generating Loss Curves...\n"
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    for column in train_loss_df.columns:
        plt.plot(train_loss_df[column], label=column)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()  # Show the legend to identify each epoch
    plt.grid(True)  # Add a grid for easier reading
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig("Results/loss_curves.png", dpi=300)
    images.append("Results/loss_curves.png")
    Q1, Q3 = np.percentile(mse_losses, [25, 75])
    Dict = {"Q1": Q1, "Q3": Q3}
    pickle.dump(Dict, open('Results/INFO.pkl', 'wb'))

  else:
    Q1, Q3 = 0.19226229563355446, 0.7454282641410828
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold_factor * IQR
    upper_bound = Q3 + threshold_factor * IQR
    # print(lower_bound, upper_bound)
    data_path = os.path.join(os.path.abspath('Data'),sample_choice)
    dataframe = pd.read_csv(data_path).sample(frac=data_slicing_percentage/100, random_state=42).reset_index(drop=True)
    anomaly_summary += f"[INFO] Saving model, scaler, Dataset Used...\n"
    dataframe.to_csv('Results/Scaled_dataset.csv', columns=dataframe.columns, index=False)
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    original_df = scaler.inverse_transform(dataframe.iloc[:,:-1])
    original_df = pd.DataFrame(original_df, columns=dataframe.columns[:-1])
    original_df['Label'] = dataframe['Label']
    original_df.to_csv('Results/Original_dataset.csv', columns=dataframe.columns, index=False)
    shutil.copy('scaler.pkl', 'Results/scaler.pkl')
    shutil.copy('model.pth', 'Results/model.pth')
    # andrew curve of dataset
    anomaly_summary += f"[INFO] Generating Andrews Curves...\n"
    make_better_andrews_curves(dataframe, 'Label',
                               colors=['Blue', 'Red'],
                               plot_title="Dataset Andrews Curves",
                               line_width=1.2,
                               transparency=0.7,
                               legend_loc='upper right',
                               figsize=(12, 7),
                               name = "Results/Dataset_andrews_curves")
    images.append("Results/Dataset_andrews_curves.png")
    inputdf = torch.tensor(dataframe.iloc[:,:-1].to_numpy(), dtype=torch.float32, device=device)
    outputdf = dataframe['Label']
    model = torch.load("model.pth",weights_only = False, map_location=device)
    model.eval()
    outputs = model(inputdf.unsqueeze(1)).squeeze(1)
    mse_loss = F.mse_loss(outputs, inputdf, reduction='none')
    mse_loss_per_data_point = mse_loss.mean(dim=-1)
    anomaly_scores = pd.DataFrame({'Loss': mse_loss_per_data_point.detach().cpu().numpy(), 'Label': outputdf})
    anomaly_scores['Anomaly'] = anomaly_scores['Loss'].apply(lambda x: 1 if x > upper_bound else 0)
    anomaly_scores['Label'] = anomaly_scores['Label'].apply(lambda x: 1 if x == "Malicious" else 0)
    out_confusion_matrix = confusion_matrix(anomaly_scores['Label'], anomaly_scores['Anomaly'])
    disp = ConfusionMatrixDisplay(confusion_matrix=out_confusion_matrix, display_labels=["Benign","Malignant"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(f"Results/confusion_matrix.png", dpi=300)
    images.append("Results/confusion_matrix.png")
    accuracy = accuracy_score(anomaly_scores['Label'], anomaly_scores['Anomaly'])
    precision = precision_score(anomaly_scores['Label'], anomaly_scores['Anomaly'])
    recall = recall_score(anomaly_scores['Label'], anomaly_scores['Anomaly'])
    f1 = f1_score(anomaly_scores['Label'], anomaly_scores['Anomaly'])
    # print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    anomaly_summary += f"[RESULT] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    anomaly_summary = anomaly_summary + f"Confusion Matrix:\n{out_confusion_matrix}\n"

  folder_path = "Results"
  with zipfile.ZipFile("Results.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, folder_path)
            zipf.write(file_path, relative_path)

  return anomaly_summary, images, "Results.zip"

iface = gr.Interface(
    fn=detect_anomalies,
    inputs=[
        gr.File(file_types=[".csv"], label="Upload CSV File"),
        gr.Radio(["Benign500.csv", "Malignant500.csv", "Balance1000.csv", "Custom Data"], value="Custom Data", label="Choose Samples or CustomData"),
        gr.Slider(minimum=10, maximum=100, step=10, value=80, label="Data Usage Percentage (Training or Detection)"),
        gr.Slider(minimum=1, maximum=20, step=1, value=3, label="Training Epochs (Default value is 3)"),
        gr.Slider(minimum=0, maximum=5, step=0.5, value=1.5, label="Loss Threshold (x, higher x means high threshold) = Q3 + x*IQR"),
    ],
    outputs=[
        gr.Textbox(label="Anomaly Summary"),
        gr.Gallery(label="Anomaly Plots"),
        "file",
    ],
<<<<<<< HEAD
    title="Your own Anomaly Detector",
    description="""
    ### Fully Unsupervised Anomaly Detection Tool (uses Bidirectional based Autoencoder with skip conn. and Dropout Layers)
    ##### Download *"Result.zip"* (contains model.pkl, dataset images, output images) to download the results from Right Bottom.
    Upload a *CSV file* (Custom Anomalies Detection: Use Output Column: "Label" or ), or Use *our trained model*.
=======
    title="Your own Anomaly Detector(LSTM based Autoencoder)",
    description="""
    ### Fully Unsupervised Anomaly Detection Tool (uses Bidirectional based Autoencoder with skip conn. and Dropout Layers)
    ##### Download *"Result.zip"* (contains model.pkl, dataset images, output images) to download the results from Right Bottom.
    Upload a *CSV file* (Custom Anomalies Detection: Use Output Column: "Label"), or Use *our trained model*.
>>>>>>> 0f1b551798e83326b79fb965a64927cbd7d89739
    """
)

if __name__ == "__main__":
    iface.launch(debug=False)