import SwiftUI
import AVFoundation
import Combine

// MARK: - ViewModel

final class RLViewModel: ObservableObject {

    // UI state
    @Published var actionImage: UIImage? = nil
    @Published var isConnected: Bool = false
    @Published var statusMessage: String = "Not connected"
    @Published var framesSent: Int = 0

    // Sub-managers
    private let cameraManager = CameraManager()
    private let wsManager = WebSocketManager()

    // Change this to your server's address
    private let serverURL = "ws://192.168.86.249:8000/ws"

    init() {
        cameraManager.delegate = self
        wsManager.delegate = self
    }

    // MARK: - Lifecycle

    func onAppear() {
        cameraManager.setup { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.cameraManager.startCapture()
                self.wsManager.connect(to: self.serverURL)
            case .failure(let error):
                DispatchQueue.main.async {
                    self.statusMessage = error.localizedDescription
                }
            }
        }
    }

    func onDisappear() {
        cameraManager.stopCapture()
        wsManager.disconnect()
    }

    // MARK: - Camera preview

    func makePreviewLayer(bounds: CGRect) -> AVCaptureVideoPreviewLayer {
        cameraManager.makePreviewLayer(for: bounds)
    }

    // MARK: - Reward buttons

    func sendReward(_ value: Float) {
        wsManager.sendReward(value)
    }
}

// MARK: - CameraManagerDelegate

extension RLViewModel: CameraManagerDelegate {
    func didCaptureFrame(_ jpegData: Data) {
        wsManager.sendFrame(jpegData)
        DispatchQueue.main.async {
            self.framesSent += 1
        }
    }
}

// MARK: - WebSocketManagerDelegate
extension RLViewModel: WebSocketManagerDelegate {
    func didReceiveAction(_ action: ActionMessage) {
        // 1. Decode the Base64 string from the 'action' field
        // Note: This happens on a background thread (from the WebSocket)
        if let imageData = Data(base64Encoded: action.action),
           let decodedImage = UIImage(data: imageData) {
            
            // 2. Switch to Main Thread for UI updates
            DispatchQueue.main.async {
                self.actionImage = decodedImage
            }
        } else {
            print("Error: Could not decode image data from action string")
        }
    }

    func didChangeConnectionState(_ connected: Bool) {
        isConnected = connected
        statusMessage = connected ? "Connected" : "Disconnected — retrying…"
        if !connected {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                guard let self else { return }
                self.wsManager.connect(to: self.serverURL)
            }
        }
    }
}

// MARK: - Camera Preview (UIViewRepresentable)

struct CameraPreview: UIViewRepresentable {
    let viewModel: RLViewModel

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // Add preview layer once the view has a non-zero frame
        guard uiView.bounds != .zero,
              uiView.layer.sublayers?.contains(where: { $0 is AVCaptureVideoPreviewLayer }) == false
        else { return }

        let previewLayer = viewModel.makePreviewLayer(bounds: uiView.bounds)
        uiView.layer.insertSublayer(previewLayer, at: 0)
    }
}

// MARK: - ContentView
struct ContentView: View {
    @StateObject private var viewModel = RLViewModel()

    var body: some View {
        ZStack {
            // ── Background ──
            Color.black.ignoresSafeArea()

            // ── LAYER 1: The Action Image (Scaled to Fit) ──
            Group {
                if let uiImage = viewModel.actionImage {
                    Image(uiImage: uiImage)
                        .interpolation(.none) // Keeps the agent's pixels crisp
                        .resizable()
                        .scaledToFit()
                } else {
                    VStack(spacing: 12) {
                        ProgressView().tint(.white)
                        Text("Connecting to Agent...")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.6))
                    }
                }
            }
            // We do NOT use ignoresSafeArea() here so it stays within the HUD bounds

            // ── LAYER 2: HUD Overlay ──
            VStack {
                // Top Status Bar
                HStack {
                    HStack{
                        Circle()
                            .fill(viewModel.isConnected ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        Text(viewModel.statusMessage)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.white)
                            .shadow(radius: 2)
                        Spacer()
                    }
                    Spacer()
                    
                    Text("Frame: \(viewModel.framesSent)")
                }
                .font(.system(size: 12, weight: .bold, design: .monospaced))
                .foregroundStyle(.white)
                .padding()
                .background(Color.black.opacity(0.5))
                
                Spacer()

                // Bottom Controls
                HStack {
                    Spacer()
                    RewardButton(label: "➕", color: .green) {
                        viewModel.sendReward(+1.0)
                    }
                    Spacer()
                }
                .padding(.bottom, 50)
            }
        }
        .onAppear { viewModel.onAppear() }
        .onDisappear { viewModel.onDisappear() }
    }
}

// MARK: - RewardButton

struct RewardButton: View {
    let label: String
    let color: Color
    let action: () -> Void

    @State private var pressed = false

    var body: some View {
        Button(action: {
            withAnimation(.easeOut(duration: 0.1)) { pressed = true }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                withAnimation { pressed = false }
            }
            action()
        }) {
            Text(label)
                .font(.system(size: 36))
                .frame(width: 72, height: 72)
                .background(color, in: Circle())
                .scaleEffect(pressed ? 0.92 : 1.0)
                // Adds a slight light-up effect when pressed
                .brightness(pressed ? 0.2 : 0)
        }
        .buttonStyle(.plain)
    }
}
// MARK: - Preview

#Preview {
    ContentView()
}
