import AVFoundation
import UIKit
import CoreImage

// MARK: - Camera Manager Implementation

protocol CameraManagerDelegate: AnyObject {
    func didCaptureFrame(_ jpegData: Data)
}

final class CameraManager: NSObject {

    weak var delegate: CameraManagerDelegate?

    var targetFPS: Double = 30.0
    var outputSize = CGSize(width: 128, height: 128)
    var jpegQuality: CGFloat = 0.6

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.johnmartin.Painter.camera.session")
    private let outputQueue = DispatchQueue(label: "com.johnmartin.Painter.camera.output")
    private let ciContext = CIContext()

    private var lastSentTime: CFTimeInterval = 0

    func setup(completion: @escaping (Result<Void, Error>) -> Void) {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self = self else { return }
            guard granted else {
                DispatchQueue.main.async { completion(.failure(CameraError.permissionDenied)) }
                return
            }
            self.sessionQueue.async {
                do {
                    try self.configureSession()
                    DispatchQueue.main.async { completion(.success(())) }
                } catch {
                    DispatchQueue.main.async { completion(.failure(error)) }
                }
            }
        }
    }

    private func configureSession() throws {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .medium

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else {
            throw CameraError.deviceUnavailable
        }
        captureSession.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        guard captureSession.canAddOutput(videoOutput) else {
            throw CameraError.outputUnavailable
        }
        captureSession.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }

        captureSession.commitConfiguration()
    }

    func startCapture() {
        sessionQueue.async { [weak self] in
            guard self?.captureSession.isRunning == false else { return }
            self?.captureSession.startRunning()
        }
    }

    func stopCapture() {
        sessionQueue.async { [weak self] in
            guard self?.captureSession.isRunning == true else { return }
            self?.captureSession.stopRunning()
        }
    }

    func makePreviewLayer(for bounds: CGRect) -> AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.frame = bounds
        layer.videoGravity = .resizeAspectFill
        return layer
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = CACurrentMediaTime()
        let interval = 1.0 / targetFPS
        guard now - lastSentTime >= interval else { return }
        lastSentTime = now

        guard let jpegData = process(sampleBuffer) else { return }

        DispatchQueue.main.async { [weak self] in
            self?.delegate?.didCaptureFrame(jpegData)
        }
    }

    private func process(_ sampleBuffer: CMSampleBuffer) -> Data? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        let scaleX = outputSize.width  / ciImage.extent.width
        let scaleY = outputSize.height / ciImage.extent.height
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let qualityKey = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        let options: [CIImageRepresentationOption: Any] = [
            qualityKey: jpegQuality
        ]
        
        return ciContext.jpegRepresentation(of: scaledImage, colorSpace: colorSpace, options: options)
    }
}

// MARK: - Errors

enum CameraError: LocalizedError {
    case permissionDenied
    case deviceUnavailable
    case outputUnavailable

    var errorDescription: String? {
        switch self {
        case .permissionDenied:  return "Camera permission denied."
        case .deviceUnavailable: return "No camera device available."
        case .outputUnavailable: return "Could not attach video output."
        }
    }
}
