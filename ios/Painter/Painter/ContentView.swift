import SwiftUI
import AVFoundation
import Combine
import UIKit

// MARK: - ViewModel

final class RLViewModel: ObservableObject {

    // UI state
    @Published var actionImage: UIImage? = nil
    @Published var isConnected: Bool = false
    @Published var statusMessage: String = "Not connected"
    @Published var framesSent: Int = 0

    // Sub-managers
    let cameraManager = CameraManager()
    let wsManager = WebSocketManager()

    // Change this to your server's address
    private let serverURL = "ws://192.168.86.249:8000/ws"

    init() {
        cameraManager.delegate = self
        wsManager.delegate = self
    }

    // MARK: - Lifecycle

    func onAppear() {
        cameraManager.setup { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success:
                // Start by establishing network state before launching camera frames
                self.statusMessage = "Connecting..."
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

// MARK: - WebSocketManagerDelegate Implementation

extension RLViewModel: WebSocketManagerDelegate {
    
    func didReceiveActionImage(_ image: UIImage, step: UInt32, row: UInt8, col: UInt8, reward: Float) {
        // Silently receive the metadata to keep the network pipeline flowing,
        // but only publish the image to the UI thread
        DispatchQueue.main.async { [weak self] in
            self?.actionImage = image
        }
    }

    func didChangeConnectionState(_ connected: Bool) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.isConnected = connected
            
            if connected {
                self.statusMessage = "Connected"
                self.cameraManager.startCapture() // Run frame streaming safely
            } else {
                self.statusMessage = "Disconnected — retrying…"
                self.cameraManager.stopCapture()
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                    guard let self = self else { return }
                    guard !self.isConnected else { return }
                    self.wsManager.connect(to: self.serverURL)
                }
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

            // ── LAYER 2: HUD Overlay ──
            VStack {
                // Top Status Bar
                HStack {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(viewModel.isConnected ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        Text(viewModel.statusMessage)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.white)
                            .shadow(radius: 2)
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
                .brightness(pressed ? 0.2 : 0)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
