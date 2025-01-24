import { Component, EventEmitter, Output, output } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { PredictGenreService } from '../../services/predict-genre.service';
import { ProbObject } from '../../models/predict-genre-response.model';
import { HttpEventType, HttpResponse } from '@angular/common/http';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { Subscription } from 'rxjs';
@Component({
  selector: 'app-file-upload',
  imports: [MatIconModule, MatProgressSpinnerModule],
  templateUrl: './file-upload.component.html',
  styleUrl: './file-upload.component.scss',
})
export class FileUploadComponent {
  uploadedFile = output<File>();

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
      this.uploadedFile.emit(file);
    }
  }
}
