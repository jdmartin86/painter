import SwiftUI
import AVFoundation

struct ContentView: View {
    var body: some View {
        CameraPreview()
            .ignoresSafeArea()
    }
}

// Wraps AVCaptureVideoPreviewLayer in a SwiftUI view
struct CameraPreview: UIViewRepresentable {
    let session = AVCaptureSession()

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        
        // Check camera permission
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        print("Camera permission status: \(status.rawValue)")
        // 0 = not determined, 1 = restricted, 2 = denied, 3 = authorized
        
        // Check camera device
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                    for: .video,
                                                    position: .front) else {
            print("ERROR: Could not find front camera")
            return view
        }
        print("Found camera: \(device.localizedName)")
        
        // Check input
        guard let input = try? AVCaptureDeviceInput(device: device) else {
            print("ERROR: Could not create camera input")
            return view
        }
        print("Camera input created successfully")
        
        session.addInput(input)
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.previewLayer = previewLayer
        view.layer.addSublayer(previewLayer)
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
            print("Session running: \(self.session.isRunning)")
        }
        
        return view
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer?.frame = uiView.bounds
    }
}

// A UIView subclass that holds a reference to the preview layer
class PreviewView: UIView {
    var previewLayer: AVCaptureVideoPreviewLayer?
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}
