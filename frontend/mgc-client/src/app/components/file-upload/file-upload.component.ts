import { Component, output } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { PredictGenreService } from '../../services/predict-genre.service';
import { ProbObject } from '../../models/predict-genre-response.model';

@Component({
  selector: 'app-file-upload',
  imports: [MatIconModule],
  templateUrl: './file-upload.component.html',
  styleUrl: './file-upload.component.scss',
})
export class FileUploadComponent {
  probObject = output<ProbObject>();

  constructor(private predictGenreService: PredictGenreService) {}

  handleChange(event: Event) {
    const input = event.target as HTMLInputElement;

    if (input?.files && input.files[0]) {
      const file = input.files[0];

      const allowedTypes = ['audio/mpeg', 'audio/wav'];
      if (!allowedTypes.includes(file.type)) {
        alert('Only MP3 and WAV files are supported.');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        alert('The file is too large! Maximum size allowed is 10 MB.');
        return;
      }
      this.predictGenreService.uploadFile(file).subscribe({
        next: (response: ProbObject) => {
          this.probObject.emit(response);
        },
        error: (error) => {
          console.error('Error while uploading the file:', error);
          alert('Failed to upload the file.');
        },
      });
    }
  }
}
